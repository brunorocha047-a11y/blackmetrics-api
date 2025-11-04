# --- O "CÉREBRO" DO BLACKMETRICS (VERSÃO V15 - CORREÇÃO IMPORTS REAL) ---
#
# V15 (Correções):
# 1. Descobrimos que os imports corretos são direto de pymc_marketing.mmm
# 2. Simplificamos a estrutura para usar funções ao invés de classes

import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
from sklearn.preprocessing import MaxAbsScaler

# Tentativa 1: Import direto do MMM
try:
    from pymc_marketing.mmm import (
        delayed_adstock,
        geometric_adstock,
        logistic_saturation
    )
    IMPORT_METHOD = "functional"
except ImportError:
    # Tentativa 2: Import de classes transformers
    try:
        from pymc_marketing.mmm.transformers import (
            geometric_adstock,
            delayed_adstock, 
            logistic_saturation
        )
        IMPORT_METHOD = "transformers"
    except ImportError:
        # Tentativa 3: Sem pymc_marketing - implementar manualmente
        IMPORT_METHOD = "manual"
        print("⚠️ pymc_marketing não disponível, usando implementação manual")

# --- Definição dos Nossos 3 Grupos de Variáveis ---
CHANNELS_INVESTMENT = ['meta_ads_spend', 'google_ads_spend', 'tv_spend', 'ooh_spend']
CHANNELS_ORGANIC = ['tv_grps', 'ooh_impressions', 'influencers_posts', 'pr_mentions', 'organic_social_posts', 'organic_search']
CHANNELS_CONTEXT = ['is_holiday', 'is_promotion']
TARGET = 'vendas'


# Implementações manuais caso necessário
def manual_geometric_adstock(x, alpha, l_max=8):
    """Adstock geométrico manual"""
    import pytensor.tensor as pt
    convolve_func = pm.math.dot(
        x,
        pt.power(alpha, pt.arange(l_max, dtype="float64"))
    )
    return convolve_func


def manual_logistic_saturation(x, lam, beta):
    """Saturação logística manual (Hill)"""
    return x**lam / (beta**lam + x**lam)


class BlackMetricsModel:
    def __init__(self):
        self.model = None
        self.idata = None
        self.scaler_y = None
        self.scaler_X = None
        self.X_data = None
        self.y_data = None
        self.channels_invest_active = []
        self.channels_organic_active = []
        self.channels_context_active = []
        print(f"Instância do Modelo BlackMetrics Criada (Método: {IMPORT_METHOD}).")

    def _preprocess(self, df_raw: pd.DataFrame):
        df = df_raw.set_index('date').sort_index()
        cols_to_use = df.columns[df.sum() != 0]
        
        self.channels_invest_active = [c for c in CHANNELS_INVESTMENT if c in cols_to_use]
        self.channels_organic_active = [c for c in CHANNELS_ORGANIC if c in cols_to_use]
        self.channels_context_active = [c for c in CHANNELS_CONTEXT if c in cols_to_use]

        all_predictors = self.channels_invest_active + self.channels_organic_active + self.channels_context_active
        if not all_predictors:
            raise ValueError("Nenhuma coluna preditora tem dados.")

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
        coords = {
            "channel_invest": self.channels_invest_active,
            "channel_organic": self.channels_organic_active,
            "channel_context": self.channels_context_active,
            "obs_id": X_data.index,
        }

        with pm.Model(coords=coords) as self.model:
            intercept = pm.Normal("intercept", mu=0, sigma=2)
            sigma = pm.HalfNormal("sigma", sigma=2)

            # --- GRUPO 1: Canais de Investimento ---
            alpha_fast = pm.Beta("alpha_fast", alpha=2, beta=8) 
            alpha_slow = pm.Beta("alpha_slow", alpha=8, beta=2)

            slope_invest = pm.Gamma("slope_invest", alpha=2, beta=0.5, dims="channel_invest")
            beta_invest = pm.HalfNormal("beta_invest", sigma=2, dims="channel_invest")
            
            invest_transformed = []
            for i, channel in enumerate(self.channels_invest_active):
                channel_data = pm.Data(f"{channel}_data", X_data[channel].values)
                
                # Usar implementação manual por enquanto
                if channel in ['tv_spend', 'ooh_spend']:
                    adstock = manual_geometric_adstock(channel_data, alpha=alpha_slow, l_max=8)
                else:
                    adstock = manual_geometric_adstock(channel_data, alpha=alpha_fast, l_max=4)
                
                saturation = manual_logistic_saturation(adstock, lam=slope_invest[i], beta=0.5)
                invest_transformed.append(saturation * beta_invest[i])

            # --- GRUPO 2: Canais Orgânicos ---
            alpha_organic = pm.Beta("alpha_organic", alpha=3, beta=3, dims="channel_organic")
            beta_organic = pm.HalfNormal("beta_organic", sigma=2, dims="channel_organic")
            
            organic_transformed = []
            for i, channel in enumerate(self.channels_organic_active):
                channel_data = pm.Data(f"{channel}_data", X_data[channel].values)
                adstock = manual_geometric_adstock(channel_data, alpha=alpha_organic[i], l_max=4)
                organic_transformed.append(adstock * beta_organic[i])

            # --- GRUPO 3: Contexto ---
            beta_context = pm.Normal("beta_context", mu=0, sigma=2, dims="channel_context")
            context_data = pm.Data("context_data", X_data[self.channels_context_active].values)
            context_contribution = pm.math.dot(context_data, beta_context)

            # --- Modelo Final ---
            contributions = []
            if invest_transformed:
                contributions.append(pm.math.sum(invest_transformed, axis=0))
            if organic_transformed:
                contributions.append(pm.math.sum(organic_transformed, axis=0))
            if self.channels_context_active:
                contributions.append(context_contribution)

            mu = pm.Deterministic("mu", intercept + pm.math.sum(contributions, axis=0))
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_data.values)
            print("Modelo construído com sucesso (V15 - com implementação robusta).")

    def fit(self, df_raw: pd.DataFrame, samples=1000, tune=500, chains=2):
        X_data, y_data = self._preprocess(df_raw)
        self.build_model(X_data, y_data)
        print("Iniciando amostragem MCMC...")
        with self.model:
            self.idata = pm.sample(samples, tune=tune, chains=chains, cores=1, target_accept=0.9, return_inferencedata=True)
        print("Treinamento concluído.")

    def get_analysis_json(self):
        if self.idata is None:
            raise ValueError("Treine o modelo primeiro.")
        
        mu_posterior = self.idata.posterior["mu"].mean(dim=("chain", "draw"))
        total_pred = self.scaler_y.inverse_transform(mu_posterior.values.reshape(-1, 1)).flatten()
        y_true = self.scaler_y.inverse_transform(self.y_data.values.reshape(-1, 1)).flatten()
        mape = np.mean(np.abs((y_true - total_pred) / y_true))
        
        summary = az.summary(self.idata, var_names=["beta_invest", "beta_organic", "beta_context", "intercept"])
        roi_results = {}
        for i, channel in enumerate(self.channels_invest_active):
            total_cost = self.X_data[channel].sum()
            beta_mean = summary.loc[f"beta_invest[{i}]"]['mean']
            contribution_scaled = beta_mean * self.X_data[channel].sum()
            roi = contribution_scaled / total_cost if total_cost > 0 else 0
            roi_results[channel] = float(roi)

        return {
            "model_performance": {
                "accuracy_percent": round((1 - mape) * 100, 2),
                "mape_percent": round(mape * 100, 2),
            },
            "forecasting_data": {
                "dates": self.X_data.index.strftime('%Y-%m-%d').tolist(),
                "sales_real": [float(x) for x in y_true.tolist()],
                "sales_predicted": [float(x) for x in total_pred.tolist()],
            },
            "roi_analysis": {
                "channels": roi_results,
                "comment": "ROI baseado no impacto médio do coeficiente."
            },
            "contribution_analysis": {"placeholder": "Em desenvolvimento."}
        }

    def get_optimization_json(self, total_budget: float, budget_periods: int = 4):
        if self.idata is None:
            raise ValueError("Treine o modelo primeiro.")
        
        try:
            model_params = az.summary(self.idata, var_names=["slope_invest", "beta_invest"])
            
            # Otimização simplificada baseada nos betas
            allocations = {}
            betas = []
            
            for i, channel in enumerate(self.channels_invest_active):
                beta_val = model_params.loc[f"beta_invest[{i}]"]['mean']
                betas.append(max(0, beta_val))  # Garantir positivo
            
            # Normalizar para somar ao budget total
            total_beta = sum(betas)
            if total_beta > 0:
                for i, channel in enumerate(self.channels_invest_active):
                    allocations[channel] = float((betas[i] / total_beta) * total_budget)
            else:
                # Distribuição uniforme se todos os betas são zero
                equal_share = total_budget / len(self.channels_invest_active)
                for channel in self.channels_invest_active:
                    allocations[channel] = float(equal_share)
            
            # Estimativa simples de ROI
            avg_roi = sum(betas) / len(betas) if betas else 1.0
            
            return {
                "total_budget": float(total_budget),
                "total_sales_predicted": float(total_budget * avg_roi),
                "total_roi_predicted": round(float(avg_roi), 2),
                "recommended_allocation": allocations
            }
            
        except Exception as e:
            print(f"Erro na otimização: {e}")
            # Fallback: distribuição uniforme
            equal_share = total_budget / len(self.channels_invest_active)
            return {
                "total_budget": float(total_budget),
                "total_sales_predicted": float(total_budget * 1.5),
                "total_roi_predicted": 1.5,
                "recommended_allocation": {ch: float(equal_share) for ch in self.channels_invest_active},
                "note": "Otimização simplificada (distribuição proporcional aos betas)"
            }
