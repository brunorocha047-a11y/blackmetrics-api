# --- O "CÉREBRO" DO BLACKMETRICS ---
# Este arquivo contém a lógica do modelo Bayesiano (MMM)
# Ele usa PyMC e PyMC-Marketing para replicar a lógica da Purple Metrics.

import pymc as pm
import pymc_marketing as pmk
import pandas as pd
import numpy as np
import arviz as az
from sklearn.preprocessing import MaxAbsScaler

# --- Definição dos Nossos 3 Grupos de Variáveis ---
# (Conforme nosso script de lógica V5)

# GRUPO 1: Canais de Investimento (Otimizáveis)
# Vamos calcular ROI, Adstock, Saturation e Otimizar
CHANNELS_INVESTMENT = [
    'meta_ads_spend', 
    'google_ads_spend', 
    'tv_spend', 
    'ooh_spend'
]

# GRUPO 2: Canais Orgânicos (Apoio)
# Vamos calcular Adstock, mas NÃO ROI e NÃO Otimizar
CHANNELS_ORGANIC = [
    'tv_grps', 
    'ooh_impressions', 
    'influencers_posts', 
    'pr_mentions', 
    'organic_social_posts', 
    'organic_search'
]

# GRUPO 3: Contexto (Fatores Externos)
# Apenas regressores de controle. SEM Adstock, SEM ROI, SEM Otimizar.
CHANNELS_CONTEXT = [
    'is_holiday', 
    'is_promotion'
]

# ALVO
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
        print("Instância do Modelo BlackMetrics Criada.")

    def _preprocess(self, df_raw: pd.DataFrame):
        """Prepara o DataFrame para o modelo."""
        
        # Garantir que a data seja o índice (necessário para Adstock)
        df = df_raw.set_index('date').sort_index()
        
        # Lidar com o "Modelo Universal" (canais vazios)
        # Se uma coluna tiver soma 0, ela não será usada.
        cols_to_use = df.columns[df.sum() != 0]
        
        # Atualiza nossas listas de canais para usar apenas os que têm dados
        self.channels_invest = [c for c in CHANNELS_INVESTMENT if c in cols_to_use]
        self.channels_organic = [c for c in CHANNELS_ORGANIC if c in cols_to_use]
        self.channels_context = [c for c in CHANNELS_CONTEXT if c in cols_to_use]

        # Separar X (preditores) e y (alvo)
        all_predictors = self.channels_invest + self.channels_organic + self.channels_context
        X = df[all_predictors]
        y = df[TARGET]

        # Escalar os dados (ajuda o MCMC a rodar mais rápido)
        # Usamos MaxAbsScaler para preservar os "zeros"
        self.scaler_X = MaxAbsScaler()
        self.scaler_y = MaxAbsScaler()

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

        # Salvar os dados para uso nos cálculos de ROI e Otimização
        self.X_data = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        self.y_data = pd.Series(y_scaled, index=y.index, name=y.name)
        
        print(f"Dados pré-processados. Colunas de investimento ativas: {self.channels_invest}")
        
        return self.X_data, self.y_data

    def build_model(self, X_data: pd.DataFrame, y_data: pd.Series):
        """Define a arquitetura do modelo PyMC."""
        
        # Coordenadas para o PyMC (ajuda a organizar os resultados)
        coords = {
            "channel_invest": self.channels_invest,
            "channel_organic": self.channels_organic,
            "channel_context": self.channels_context,
            "obs_id": X_data.index,
        }

        with pm.Model(coords=coords) as self.model:
            
            # --- Definição do Modelo ---
            
            # 1. Intercepto (Vendas Base)
            intercept = pm.Normal("intercept", mu=0, sigma=2)

            # 2. Sigma (Ruído/Erro do Modelo)
            sigma = pm.HalfNormal("sigma", sigma=2)

            # --- GRUPO 1: Canais de Investimento (com Priors e Adstock Avançado) ---
            
            # Priors de Adstock Rápido (Google/Meta)
            # Força o decaimento (alpha) a ser baixo (Beta(2, 8))
            alpha_fast = pm.Beta("alpha_fast", alpha=2, beta=8) 
            
            # Priors de Adstock Lento (TV/OOH)
            # Força o decaimento (alpha) a ser alto (Beta(8, 2))
            alpha_slow = pm.Beta("alpha_slow", alpha=8, beta=2)
            # Força o atraso do pico (theta) a ser maior que zero (Gamma)
            theta_slow = pm.Gamma("theta_slow", alpha=2, beta=1) 

            # Priors de Saturação (para todos do Grupo 1)
            slope_invest = pm.Gamma("slope_invest", alpha=2, beta=0.5, dims="channel_invest")
            
            # Coeficientes (Impacto) - Forçados a serem positivos (HalfNormal)
            beta_invest = pm.HalfNormal("beta_invest", sigma=2, dims="channel_invest")
            
            # Aplicar as transformações
            invest_transformed = []
            for channel in self.channels_invest:
                channel_data = pm.Data(f"{channel}_data", X_data[channel], mutable=True)
                
                if channel in ['tv_spend', 'ooh_spend']:
                    # Aplicar Adstock Lento/Atrasado (Delayed)
                    adstock = pmk.DelayedAdstock(channel_data, alpha=alpha_slow, theta=theta_slow, max_lag=8)
                else:
                    # Aplicar Adstock Rápido (Geometric)
                    adstock = pmk.GeometricAdstock(channel_data, alpha=alpha_fast, max_lag=4)
                
                # Aplicar Saturação (Hill)
                saturation = pmk.Hill(adstock, slope=slope_invest[channel], k=0.5)
                invest_transformed.append(saturation * beta_invest[channel])

            # --- GRUPO 2: Canais Orgânicos (com Adstock Simples) ---
            # (Não aplicamos Saturação de custo ou Otimização)
            
            # Usamos Adstock Geométrico simples para todos
            alpha_organic = pm.Beta("alpha_organic", alpha=3, beta=3, dims="channel_organic")
            beta_organic = pm.HalfNormal("beta_organic", sigma=2, dims="channel_organic")
            
            organic_transformed = []
            for i, channel in enumerate(self.channels_organic):
                channel_data = pm.Data(f"{channel}_data", X_data[channel], mutable=True)
                adstock = pmk.GeometricAdstock(channel_data, alpha=alpha_organic[i], max_lag=4)
                organic_transformed.append(adstock * beta_organic[i])

            # --- GRUPO 3: Contexto (Regressores Simples) ---
            # (Podem ser positivos ou negativos - pm.Normal)
            
            beta_context = pm.Normal("beta_context", mu=0, sigma=2, dims="channel_context")
            context_data = pm.Data("context_data", X_data[self.channels_context], mutable=True)
            context_contribution = pm.math.dot(context_data, beta_context)

            # --- Modelo Final (μ) ---
            # μ = Base + Grupo 1 + Grupo 2 + Grupo 3
            mu = pm.Deterministic(
                "mu",
                intercept + 
                pm.math.sum(invest_transformed, axis=0) + 
                pm.math.sum(organic_transformed, axis=0) + 
                context_contribution
            )

            # --- Likelihood (Observador) ---
            # Onde o modelo compara a previsão (mu) com a realidade (y_data)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_data)
            
            print("Arquitetura do Modelo PyMC construída com sucesso.")

    def fit(self, df_raw: pd.DataFrame, samples=2000, tune=1000, chains=4):
        """Treina o modelo MCMC."""
        
        X_data, y_data = self._preprocess(df_raw)
        self.build_model(X_data, y_data)
        
        print("Iniciando amostragem MCMC (Isso pode levar alguns minutos)...")
        with self.model:
            # pm.sample() é o "cérebro" do MCMC.
            # Usamos 'cores=1' para compatibilidade com o plano gratuito do Render
            self.idata = pm.sample(samples, tune=tune, chains=chains, cores=1, target_accept=0.9)
        print("Treinamento MCMC concluído.")

    def get_analysis_json(self):
        """Extrai todos os resultados e os formata em um JSON para o Lovable."""
        
        if self.idata is None:
            raise ValueError("O modelo ainda não foi treinado. Chame .fit() primeiro.")

        # --- 1. Calcular Contribuição por Canal ---
        # (Usando o 'idata' que acabamos de treinar)
        mu_posterior = self.idata.posterior["mu"].mean(dim=("chain", "draw"))
        
        # Des-escalar
        total_pred = self.scaler_y.inverse_transform(mu_posterior.values.reshape(-1, 1)).flatten()
        y_true = self.scaler_y.inverse_transform(self.y_data.values.reshape(-1, 1)).flatten()
        
        # Acurácia do Forecasting
        mape = np.mean(np.abs((y_true - total_pred) / y_true))
        accuracy = 1 - mape

        # Contribuição de cada canal (o "motor" do gráfico de pizza)
        channel_contributions = {}
        # ... (lógica complexa de extração de contribuição do PyMC)
        # Por simplicidade, vamos usar as funções do pmk
        
        # NOTA: O cálculo de contribuição detalhado é complexo.
        # Para esta V1 da API, vamos nos concentrar nos coeficientes e ROI.
        
        # --- 2. Extrair Coeficientes (Impacto) ---
        summary = az.summary(self.idata, var_names=["beta_invest", "beta_organic", "beta_context", "intercept"])
        
        # --- 3. Calcular ROI (para Grupo 1) ---
        # ROI = (Contribuição Total em Vendas) / (Custo Total)
        roi_results = {}
        for i, channel in enumerate(self.channels_invest):
            # Custo Total (escalado)
            total_cost = self.X_data[channel].sum()
            
            # Coeficiente (impacto) médio
            # (Esta é uma simplificação; um cálculo de ROI completo usaria a contribuição total)
            beta_mean = summary.loc[f"beta_invest[{channel}]"]['mean']
            
            # Contribuição estimada
            # (Simplificado: impacto * gasto)
            contribution = beta_mean * total_cost 
            
            # ROI (Simplificado)
            roi = (contribution / total_cost) if total_cost > 0 else 0
            roi_results[channel] = roi

        # --- 4. Formatar o JSON de Resposta ---
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
                # (Aqui entrariam os dados do gráfico de pizza)
                "placeholder": "Módulo de contribuição em desenvolvimento."
            }
        }
        
        print("JSON de Análise gerado.")
        return response

    def get_optimization_json(self, total_budget: float, budget_periods: int = 4):
        """
        Executa o "Otimizador Proativo" (a feature do Script 1).
        """
        if self.idata is None:
            raise ValueError("O modelo ainda não foi treinado. Chame .fit() primeiro.")

        # 1. Extrair os parâmetros do modelo treinado
        # (slope e k das curvas de saturação Hill)
        model_params = az.summary(self.idata, var_names=["slope_invest"])
        
        # Mapear os parâmetros para os canais corretos
        param_map = {}
        for i, channel in enumerate(self.channels_invest):
            param_map[channel] = {
                'slope': model_params.loc[f"slope_invest[{channel}]"]['mean'],
                'k': 0.5 # (Fixamos em 0.5 na definição do modelo)
            }
        
        # 2. Definir a função de canal (Hill) que o otimizador usará
        def budget_response_function(budget, slope, k):
            return pmk.Hill(budget, slope=slope, k=k).eval()

        # 3. Chamar o otimizador do PyMC-Marketing
        # Esta função faz a mágica: encontra a alocação ideal para o budget
        try:
            optimal_allocation = pmk.optimize_budget(
                total_budget=total_budget,
                channels=self.channels_invest,
                parameters=param_map,
                response_function=budget_response_function,
                budget_unit_cost=np.ones(len(self.channels_invest)) # Custo unitário é 1 (R$ 1)
            )
            
            # 4. Calcular o resultado da otimização
            optimized_budget_series = pd.Series(optimal_allocation, index=self.channels_invest)
            
            total_response = 0
            for channel in self.channels_invest:
                params = param_map[channel]
                total_response += budget_response_function(optimized_budget_series[channel], **params)

            # Des-escalar a resposta (vendas)
            total_sales_predicted = self.scaler_y.inverse_transform([[total_response]])[0][0]
            total_sales_predicted_weekly = total_sales_predicted / budget_periods # Média semanal
            
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
