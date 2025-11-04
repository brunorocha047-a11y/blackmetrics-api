# --- O "CÉREBRO" DO BLACKMETRICS (VERSÃO V11 - COMPATÍVEL COM pymc-marketing==0.5.0) ---

import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
from sklearn.preprocessing import MaxAbsScaler

# ✅ IMPORTS CORRETOS PARA v0.5.0
from pymc_marketing.mmm import (
    GeometricAdstock,
    WeibullAdstock,
    Hill,
    optimize_budget
)

# --- Definição dos Nossos 3 Grupos de Variáveis ---
CHANNELS_INVESTMENT = ['meta_ads_spend', 'google_ads_spend', 'tv_spend', 'ooh_spend']
CHANNELS_ORGANIC = ['tv_grps', 'ooh_impressions', 'influencers_posts', 'pr_mentions', 'organic_social_posts', 'organic_search']
CHANNELS_CONTEXT = ['is_holiday', 'is_promotion']
TARGET = 'vendas'


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
        print("Instância do Modelo BlackMetrics Criada.")

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
            theta_slow = pm.Gamma("theta_slow", alpha=2, beta=1) 

            slope_invest = pm.Gamma("slope_invest", alpha=2, beta=0.5, dims="channel_invest")
            beta_invest = pm.HalfNormal("beta_invest", sigma=2, dims="channel_invest")
            
            invest_transformed = []
            for i, channel in enumerate(self.channels_invest_active):
                channel_data = pm.Data(f"{channel}_data", X_data[channel].values)
                
                if channel in ['tv_spend', 'ooh_spend']:
                    adstock = WeibullAdstock(alpha=alpha_slow, lam=theta_slow, l_max=8)(channel_data)
                else:
                    adstock = GeometricAdstock(alpha=alpha_fast, l_max=4)(channel_data)
                
                saturation = Hill(slope=slope_invest[i], k=0.5)(adstock)
                invest_transformed.append(saturation * beta_invest[i])

            # --- GRUPO 2: Canais Orgânicos ---
            alpha_organic = pm.Beta("alpha_organic", alpha=3, beta=3, dims="channel_organic")
            beta_organic = pm.HalfNormal("beta_organic", sigma=2, dims="channel_organic")
            
            organic_transformed = []
            for i, channel in enumerate(self.channels_organic_active):
                channel_data = pm.Data(f"{channel}_data", X_data[channel].values)
                adstock = GeometricAdstock(alpha=alpha_organic[i], l_max=4)(channel_data)
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
            print("Modelo construído com sucesso (V11 - compatível com pymc-marketing 0.5.0).")

    def fit(self, df_raw: pd.DataFrame, samples=1000, tune=500, chains=2):
        # Use amostras menores para teste rápido; aumente depois
        X_data, y_data = self._preprocess(df_raw)
        self.build_model(X_data, y_data)
        print("Iniciando amostragem MCMC...")
        with self.model:
            self.idata = pm.sample(samples, tune=tune, chains=chains, cores=1, target_accept=0.9)
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
            roi = beta_mean if total_cost > 0 else 0
            roi_results[channel] = roi

        return {
            "model_performance": {
                "accuracy_percent": round((1 - mape) * 100, 2),
                "mape_percent": round(mape * 100, 2),
            },
            "forecasting_data": {
                "dates": self.X_data.index.strftime('%Y-%m-%d').tolist(),
                "sales_real": y_true.tolist(),
                "sales_predicted": total_pred.tolist(),
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
        
        model_params = az.summary(self.idata, var_names=["slope_invest"])
        param_map = {
            channel: {
                'slope': model_params.loc[f"slope_invest[{i}]"]['mean'],
                'k': 0.5
            }
            for i, channel in enumerate(self.channels_invest_active)
        }
        
        def response_fn(budget, slope, k):
            return Hill(slope=slope, k=k)(budget).eval()

        try:
            optimal = optimize_budget(
                total_budget=total_budget,
                channels=self.channels_invest_active,
                parameters=param_map,
                response_function=response_fn,
                budget_unit_cost=np.ones(len(self.channels_invest_active))
            )
            
            optimized = pd.Series(optimal, index=self.channels_invest_active)
            total_resp = sum(response_fn(optimized[ch], **param_map[ch]) for ch in self.channels_invest_active)
            total_sales = self.scaler_y.inverse_transform([[total_resp]])[0][0] / budget_periods
            roi = total_sales / total_budget if total_budget > 0 else 0

            return {
                "total_budget": total_budget,
                "total_sales_predicted": total_sales,
                "total_roi_predicted": round(roi, 2),
                "recommended_allocation": optimized.to_dict()
            }
        except Exception as e:
            return {"error": "Otimização falhou", "details": str(e)}
