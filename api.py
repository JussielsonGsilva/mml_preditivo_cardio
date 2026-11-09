# Literal: permite dizer "este campo só aceita estes valores exatos"
from typing import Literal

# joblib: carrega o modelo salvo em arquivo
import joblib

# Pandas: o modelo recebe os dados no formato de tabela (DataFrame)
import pandas as pd

# Importa a classe principal do framework
from fastapi import FastAPI

# StaticFiles: permite a API entregar arquivos prontos (HTML, CSS, JS) de uma pasta
from fastapi.staticfiles import StaticFiles

# BaseModel: base para criar a "ficha" de dados; Field: regras de cada campo
from pydantic import BaseModel, Field

# O modelo salvo depende desta classe: ela PRECISA ser importada
# antes do joblib.load, senão o carregamento falha
from preprocessamento import TratadorColesterol  # noqa: F401

# ---------- Carregamento do modelo (uma única vez, quando a API sobe) ----------

# Lê o pacote salvo pelo script 19_salvar_modelo.py
PACOTE = joblib.load("modelos/modelo_cardio.joblib")

# Separa a esteira treinada e o limiar de decisão (0.30)
ESTEIRA = PACOTE["esteira"]
LIMIAR = PACOTE["limiar"]

# ---------- Tradução: campos da API (português) -> colunas do modelo (inglês) ----------

# Nome do campo na API -> nome da coluna usada no treino
CAMPOS = {
    "idade": "Age",
    "sexo": "Sex",
    "tipo_dor_peito": "ChestPainType",
    "pressao_repouso": "RestingBP",
    "colesterol": "Cholesterol",
    "glicemia_jejum_alta": "FastingBS",
    "ecg_repouso": "RestingECG",
    "freq_cardiaca_max": "MaxHR",
    "angina_exercicio": "ExerciseAngina",
    "depressao_st": "Oldpeak",
    "inclinacao_st": "ST_Slope"}

# Valores traduzidos: valor na API -> valor usado no treino
VALORES = {
    "angina_exercicio": {"S": "Y", "N": "N"},
    "inclinacao_st": {"Subindo": "Up", "Plano": "Flat", "Descendo": "Down"}}

# Guia de siglas em Markdown: aparece no topo da página /docs
GUIA_SIGLAS = """
Estima o risco de doença cardíaca a partir de dados clínicos.
**Projeto de estudo: NÃO substitui avaliação médica.**

### Guia de siglas

**tipo_dor_peito** — tipo de dor no peito

| Sigla | Significado |
|---|---|
| ASY | Assintomático (sem dor no peito) |
| ATA | Angina atípica |
| NAP | Dor não anginosa |
| TA | Angina típica |

**ecg_repouso** — resultado do eletrocardiograma em repouso

| Sigla | Significado |
|---|---|
| Normal | Eletrocardiograma normal |
| ST | Anormalidade na onda ST-T |
| LVH | Hipertrofia ventricular esquerda |

**Outros campos**

| Campo | Valores |
|---|---|
| sexo | M = masculino, F = feminino |
| glicemia_jejum_alta | 1 = glicemia em jejum acima de 120 mg/dl, 0 = normal |
| angina_exercicio | S = sim, N = não |
| inclinacao_st | Subindo, Plano ou Descendo (segmento ST no pico do exercício) |
| colesterol | 0 = exame não realizado |
"""

# Cria a aplicação; a descrição (guia de siglas) aparece no topo de /docs
app = FastAPI(
    title="API de Risco Cardíaco",
    description=GUIA_SIGLAS,
    version="1.0.0")

# Entrega os arquivos da pasta "web" no endereço /app;
# html=True faz /app abrir automaticamente o web/index.html
app.mount("/app", StaticFiles(directory="web", html=True), name="web")


# "Ficha" do paciente: cada atributo é um campo do JSON recebido.
# Field define limites (ge = maior ou igual, le = menor ou igual),
# a descrição e um exemplo que aparece pré-preenchido em /docs
class Paciente(BaseModel):

    idade: int = Field(ge=1, le=120, description="Idade em anos", examples=[54])

    sexo: Literal["M", "F"] = Field(description="M = masculino, F = feminino", examples=["M"])

    tipo_dor_peito: Literal["ASY", "ATA", "NAP", "TA"] = Field(
        description="ASY = assintomático, ATA = angina atípica, "
                    "NAP = dor não anginosa, TA = angina típica",
        examples=["ASY"])

    pressao_repouso: int = Field(ge=50, le=250, description="Pressão arterial em repouso (mm Hg)", examples=[140])

    colesterol: int = Field(ge=0, le=700, description="Colesterol sérico (mg/dl); 0 = exame não realizado", examples=[289])

    glicemia_jejum_alta: Literal[0, 1] = Field(description="1 = glicemia em jejum acima de 120 mg/dl, 0 = normal", examples=[0])

    ecg_repouso: Literal["Normal", "ST", "LVH"] = Field(
        description="Normal = normal, ST = anormalidade na onda ST-T, "
                    "LVH = hipertrofia ventricular esquerda",
        examples=["Normal"])

    freq_cardiaca_max: int = Field(ge=50, le=250, description="Frequência cardíaca máxima atingida (bpm)", examples=[122])

    angina_exercicio: Literal["S", "N"] = Field(description="Angina induzida por exercício: S = sim, N = não", examples=["S"])

    depressao_st: float = Field(ge=-5, le=10, description="Depressão do segmento ST no exercício (Oldpeak)", examples=[1.5])

    inclinacao_st: Literal["Subindo", "Plano", "Descendo"] = Field(
        description="Inclinação do segmento ST no pico do exercício", examples=["Plano"])


# Rota raiz: confirma que a API está no ar
@app.get("/")
def raiz():

    # O dicionário devolvido vira JSON automaticamente
    return {"status": "API no ar",
            "documentacao": "/docs"}


# Rota de previsão: recebe a ficha do paciente e devolve o risco
@app.post("/prever")
def prever(paciente: Paciente):

    # Se chegou aqui, o FastAPI já validou todos os campos;
    # model_dump() transforma a ficha em dicionário
    dados = paciente.model_dump()

    # Monta um novo dicionário já com os nomes e valores do modelo
    linha = {}
    for campo_pt, valor in dados.items():

        # Se o campo tem valores traduzidos, troca o valor (ex: "S" -> "Y")
        if campo_pt in VALORES:
            valor = VALORES[campo_pt][valor]

        # Grava com o nome da coluna do modelo (ex: "idade" -> "Age")
        linha[CAMPOS[campo_pt]] = valor

    # O modelo espera uma tabela: cria um DataFrame com UMA linha (o paciente)
    tabela = pd.DataFrame([linha])

    # Probabilidade de doença: [0, 1] = primeira (única) linha, coluna "com doença"
    probabilidade = float(ESTEIRA.predict_proba(tabela)[0, 1])

    # Aplica o limiar: 30% ou mais = risco alto
    risco_alto = probabilidade >= LIMIAR

    # Resposta em JSON
    return {
        "probabilidade_doenca": round(probabilidade * 100, 1),
        "risco": "ALTO" if risco_alto else "BAIXO",
        "limiar_usado": LIMIAR * 100,
        "interpretacao": ("Probabilidade igual ou acima do limiar de 30%: o modelo classifica como risco alto."
                          if risco_alto else
                          "Probabilidade abaixo do limiar de 30%: o modelo classifica como risco baixo."),
        "aviso": "Projeto de estudo. NÃO substitui avaliação médica."}
