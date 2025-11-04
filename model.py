# --- O "CÉREBRO" DO BLACKMETRICS (VERSÃO V9 - CORREÇÃO DEFINITIVA) ---
#
# V9 (Correções):
# 1. Usando a lógica do "Amigo V6" (pmm.mmm.logistic_saturation, etc.)
# 2. Aplicada a correção do "V8": pm.Data() é a função correta, mas
#    REMOVEMOS o argumento 'dims="obs_id"' para evitar o erro final de sintaxe.

import pymc as pm
import pymc_marketing as pmm
import pandas as pd
import numpy as np
import arviz as az
from sklearn.preprocessing import MaxAbsScaler
# Não precisamos do 'import pytensor.tensor as pt', ele é desnecessário

# --- Definição dos Nossos 3 Grupos de Variáveis ---
CHANNELS_INVESTMENT = [
    'meta_ads_spend', 
    'google_ads_spend', 
    'tv_spend', 
    'ooh_spend'
]
CHANNELS_ORGANIC = [
    'tv_grps', 
    'ooh_impressions', 
    'influencers_posts', 
    'pr_mentions', 
    'organic_social_posts', 
    'organic_search'
]
CHANNELS_CONTEXT = [
    'is_holiday', 
    'is_promotion'
]
TARGET = 'vendas'


class BlackMetricsModel:
    """
    Classe que encapsula o nosso modelo MMM Bayesiano.
    Ela será instanciada e chamada pela nossa API (FastAPI).
    """
    def __init__(self):
        self.model = None
        self.idata = None
        self.scaler_y = None
        self.scaler_X = None
        self.X_data = None
        self.y_data = None
        # Manter o controle das colunas ativas
        self.channels_invest_active = []
        self.channels_organic_active = []
        self.channels_context_active = []
        print("Instância do Modelo BlackMetrics Criada.")

    def _preprocess(self, df_raw: pd.DataFrame):
        """Prepara o DataFrame para o modelo."""
        
        df = df_raw.set_index('date').sort_index()
        
        cols_to_use = df.columns[df.sum() != 0]
        
        self.channels_invest_active = [c for c in CHANNELS_INVESTMENT if c in cols_to_use]
        self.channels_organic_active = [c for c in CHANNELS_ORGANIC if c in cols_to_use]
        self.channels_context_active = [c for c in CHANNELS_CONTEXT if c in cols_to_use]

        all_predictors = self.channels_invest_active + self.channels_organic_active + self.channels_context_active
        
        if not all_predictors:
             raise ValueError("Nenhuma coluna preditora tem dados (soma diferente de zero).")
        
        X = df[all_predictors]
        y = df[TARGET]

        self.scaler_X = MaxAbsScaler()
        self.scaler_y = MaxAbsScaler()

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

        self.X_data = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        self.y_data = pd.Series(y_scaled, index=y.index, name=y.name)
        
        print(f"Dados pré-processados. Colunas de investimento ativas: {self.channels_invest_active}")
        
        return self.X_data, self.y_data

    def build_model(self, X_data: pd.DataFrame, y_data: pd.Series):
        """Define a arquitetura do modelo PyMC."""
        
        coords = {
            "channel_invest": self.channels_invest_active,
            "channel_organic": self.channels_organic_active,
            "channel_context": self.channels_context_active,
            "obs_id": X_data.index,
        }

        with pm.Model(coords=coords) as self.model:
            
            # 1. Intercepto (Vendas Base)
            intercept = pm.Normal("intercept", mu=0, sigma=2)

            # 2. Sigma (Ruído/Erro do Modelo)
            sigma = pm.HalfNormal("sigma", sigma=2)

            # --- GRUPO 1: Canais de Investimento (com Priors e Adstock Avançado) ---
            alpha_fast = pm.Beta("alpha_fast", alpha=2, beta=8) 
            alpha_slow = pm.Beta("alpha_slow", alpha=8, beta=2)
            theta_slow = pm.Gamma("theta_slow", alpha=2, beta=1) 

            slope_invest = pm.Gamma("slope_invest", alpha=2, beta=0.5, dims="channel_invest")
            beta_invest = pm.HalfNormal("beta_invest", sigma=2, dims="channel_invest")
            
            invest_transformed = []
            for channel in self.channels_invest_active:
                # CORREÇÃO V9: Usar pm.Data() e remover o 'dims'
                channel_data = pm.Data(f"{channel}_data", X_data[channel].values)
                
                if channel in ['tv_spend', 'ooh_spend']:
                    adstock = pmm.mmm.delayed_adstock(channel_data, alpha=alpha_slow, theta=theta_slow, l_max=8)
                else:
                    adstock = pmm.mmm.geometric_adstock(channel_data, alpha=alpha_fast, l_max=4)
                
                saturation = pmm.mmm.logistic_saturation(
                    adstock, 
                    lam=slope_invest[self.channels_invest_active.index(channel)], 
                    beta=0.5
                )
                invest_transformed.append(saturation * beta_invest[self.channels_invest_active.index(channel)])

            # --- GRUPO 2: Canais Orgânicos (com Adstock Simples) ---
            alpha_organic = pm.Beta("alpha_organic", alpha=3, beta=3, dims="channel_organic")
            beta_organic = pm.HalfNormal("beta_organic", sigma=2, dims="channel_organic")
            
            organic_transformed = []
            for i, channel in enumerate(self.channels_organic_active):
                # CORREÇÃO V9: Usar pm.Data() e remover o 'dims'
                channel_data = pm.Data(f"{channel}_data", X_data[channel].values)
                
                adstock = pmm.mmm.geometric_adstock(
                    channel_data, 
                    alpha=alpha_organic[i], 
                    l_max=4
                )
                organic_transformed.append(adstock * beta_organic[i])

            # --- GRUPO 3: Contexto (Regressores Simples) ---
            beta_context = pm.Normal("beta_context", mu=0, sigma=2, dims="channel_context")
            # CORREÇÃO V9: Usar pm.Data() e remover o 'dims'
            context_data = pm.Data("context_data", X_data[self.channels_context_active].values)
            context_contribution = pm.math.dot(context_data, beta_context)

            # --- Modelo Final (μ) ---
            contributions = []
            if invest_transformed:
                contributions.append(pm.math.sum(invest_transformed, axis=0))
            if organic_transformed:
                contributions.append(pm.math.sum(organic_transformed, axis=0))
            if self.channels_context_active:
                contributions.append(context_contribution)

            mu = pm.Deterministic(
                "mu",
                intercept + pm.math.sum(contributions, axis=0)
            )

            # --- Likelihood (Observador) ---
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_data.values)
            
            print("Arquitetura do Modelo PyMC (V9 Definitiva) construída com sucesso.")

    def fit(self, df_raw: pd.DataFrame, samples=2000, tune=1000, chains=4):
        """Treina o modelo MCMC."""
        
        X_data, y_data = self._preprocess(df_raw)
        self.build_model(X_data, y_data)
        
        print("Iniciando amostragem MCMC (Isso pode levar alguns minutos)...")
        with self.model:
            self.idata = pm.sample(samples, tune=tune, chains=chains, cores=1, target_accept=0.9)
        print("Treinamento MCMC concluído.")

    def get_analysis_json(self):
        """Extrai todos os resultados e os formata em um JSON para o Lovable."""
        
        if self.idata is None:
            raise ValueError("O modelo ainda não foi treinado. Chame .fit() primeiro.")

        mu_posterior = self.idata.posterior["mu"].mean(dim=("chain", "draw"))
        
        total_pred = self.scaler_y.inverse_transform(mu_posterior.values.reshape(-1, 1)).flatten()
        y_true = self.scaler_y.inverse_transform(self.y_data.values.reshape(-1, 1)).flatten()
        
        mape = np.mean(np.abs((y_true - total_pred) / y_true))
        accuracy = 1 - mape

        summary = az.summary(self.idata, var_names=["beta_invest", "beta_organic", "beta_context", "intercept"])
        
        roi_results = {}
        for channel in self.channels_invest_active:
            total_cost = self.X_data[channel].sum()
            beta_mean = summary.loc[f"beta_invest[{channel}]"]['mean']
            contribution = beta_mean * total_cost 
            roi = (contribution / total_cost) if total_cost > 0 else 0
            roi_results[channel] = roi

        response = {
            "model_performance": {
                "accuracy_percent": round(accuracy * 100, 2),
                "mape_percent": round(mape * 100, 2),
            },
            "forecasting_data": {
                "dates": self.X_data.index.strftime('%Y-%m-%d').tolist(),
                "sales_real": y_true.tolist(),
                "sales_predicted": total_pred.tolist(),
            },
            "roi_analysis": {
                "channels": roi_results,
                "comment": "ROI baseado no impacto médio do coeficiente vs. custo total."
            },
            "contribution_analysis": {
                "placeholder": "Módulo de contribuição em desenvolvimento."
            }
        }
        
        print("JSON de Análise gerado.")
        return response

    def get_optimization_json(self, total_budget: float, budget_periods: int = 4):
        """
        Executa o "Otimizador Proativo".
        """
        if self.idata is None:
            raise ValueError("O modelo ainda não foi treinado. Chame .fit() primeiro.")

        model_params = az.summary(self.idata, var_names=["slope_invest"])
        
        param_map = {}
        for channel in self.channels_invest_active:
            param_map[channel] = {
                'slope': model_params.loc[f"slope_invest[{channel}]"]['mean'],
                'k': 0.5 
            }
        
        def budget_response_function(budget, slope, k):
            # Usando a função de saturação logística do pymc-marketing
            return pmm.mmm.logistic_saturation(budget, lam=slope, beta=k).eval()

        try:
            # Nota: optimize_budget pode não estar disponível em todas as versões
            # Se não funcionar, será necessário implementar otimização customizada
            optimal_allocation = pmm.budget_optimizer.optimize_budget_allocation(
                total_budget=total_budget,
                channels=self.channels_invest_active,
                parameters=param_map,
                response_function=budget_response_function
            )
            
            optimized_budget_series = pd.Series(optimal_allocation, index=self.channels_invest_active)
            
            total_response = 0
            for channel in self.channels_invest_active:
                params = param_map[channel]
                total_response += budget_response_function(optimized_budget_series[channel], **params)

            total_sales_predicted = self.scaler_y.inverse_transform([[total_response]])[0][0]
            total_sales_predicted_weekly = total_sales_predicted / budget_periods 
            
            roi = (total_sales_predicted / total_budget) if total_budget > 0 else 0

            response = {
                "total_budget": total_budget,
                "total_sales_predicted": total_sales_predicted_weekly,
                "total_roi_predicted": round(roi, 2),
                "recommended_allocation": optimized_budget_series.to_dict()
            }
            
        except Exception as e:
            print(f"Erro na otimização: {e}")
            response = {"error": "Não foi possível otimizar o budget. Verifique os parâmetros do modelo.", "details": str(e)}

        print("JSON de Otimização gerado.")
        return response
