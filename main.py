# --- A API WEB (FastAPI) ---
# Este arquivo é o "Rosto" do nosso Cérebro.
# Ele cria os endpoints (/analyze, /optimize) que o Lovable irá chamar.
# Ele será iniciado pelo Render.com.

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

# Importa nossa lógica de modelo do outro arquivo
from model import BlackMetricsModel

# --- Configuração da App ---

app = FastAPI(
    title="BlackMetrics API",
    description="O 'Cérebro' Bayesiano por trás do BlackMetrics, rodando em PyMC."
)

# Configurar CORS (Permite que o Lovable chame esta API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens (incluindo Lovable)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gerenciamento do Modelo ---
# Criamos UMA instância global do modelo.
# Isso garante que, após o treino, o modelo (com seus dados)
# permaneça na memória para ser usado pelo endpoint /optimize.
# Esta é a forma mais simples de gerenciamento de estado para esta V1.
model_brain = BlackMetricsModel()


# --- Endpoints da API ---

@app.get("/")
def read_root():
    """Endpoint raiz para verificar se a API está online."""
    return {"status": "BlackMetrics API 'Cérebro' está online."}

@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    """
    Endpoint principal. Recebe o CSV do Lovable, treina o modelo
    e retorna o JSON de análise completo.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Arquivo inválido. Por favor, envie um CSV.")

    try:
        # Ler o conteúdo do arquivo
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), parse_dates=['date'])

        # Validar colunas (simplificado)
        if 'vendas' not in df.columns or 'date' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV inválido. Colunas 'date' e 'vendas' são obrigatórias.")
            
        print("CSV recebido. Iniciando treinamento do modelo...")
        
        # Treinar o modelo (a parte demorada)
        # Na V1, este chamado é síncrono. O Lovable terá que esperar.
        global model_brain
        model_brain.fit(df)
        
        print("Treinamento concluído. Gerando JSON de resposta...")
        
        # Obter os resultados
        analysis_json = model_brain.get_analysis_json()
        
        return analysis_json

    except Exception as e:
        print(f"ERRO DE ANÁLISE: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar o arquivo: {e}")

@app.post("/optimize")
def optimize_budget(total_budget: float, budget_periods: int = 4):
    """
    Endpoint do Otimizador. Usa o modelo JÁ TREINADO na memória
    para calcular a alocação de budget ideal.
    """
    global model_brain
    
    if model_brain.idata is None:
        raise HTTPException(status_code=400, detail="O modelo ainda não foi treinado. Por favor, rode a /analyze primeiro.")
    
    try:
        print(f"Otimização solicitada para budget: {total_budget}")
        optimization_json = model_brain.get_optimization_json(
            total_budget=total_budget, 
            budget_periods=budget_periods
        )
        return optimization_json
        
    except Exception as e:
        print(f"ERRO DE OTIMIZAÇÃO: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno ao otimizar o budget: {e}")
