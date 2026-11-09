# Biblioteca padrão para lidar com pastas
import os

# joblib: salva e carrega objetos Python (como o modelo treinado) em arquivo
import joblib

# Pandas para ler os dados
import pandas as pd

# scikit-learn: usado aqui só para registrar a versão no arquivo salvo
import sklearn

# Peças da esteira
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Métricas para conferir o resultado
from sklearn.metrics import recall_score, precision_score, accuracy_score

# Nosso transformador do colesterol (arquivo preprocessamento.py)
from preprocessamento import TratadorColesterol

# Limiar escolhido na seção 6.22
LIMIAR = 0.30

# Caminho do arquivo do modelo salvo
CAMINHO_MODELO = "modelos/modelo_cardio.joblib"

# ---------- Dados no formato ORIGINAL ----------

# treino.csv e teste.csv ainda têm as colunas de texto e os zeros do colesterol,
# exatamente como os dados que a API vai receber
treino = pd.read_csv("dados/treino.csv")
teste = pd.read_csv("dados/teste.csv")

# Separa entradas (X) e resposta certa (y)
X_treino = treino.drop(columns="HeartDisease")
y_treino = treino["HeartDisease"]
X_teste = teste.drop(columns="HeartDisease")
y_teste = teste["HeartDisease"]

# ---------- Montagem da esteira completa ----------

# Colunas numéricas contínuas (serão padronizadas)
continuas = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

# Colunas de texto (serão convertidas com one-hot)
categoricas = ["Sex", "ExerciseAngina", "ChestPainType", "RestingECG", "ST_Slope"]

# Preparação das colunas:
# - StandardScaler nas contínuas
# - OneHotEncoder nas de texto; drop="if_binary" deixa só UMA coluna
#   para as binárias (Sex_M, ExerciseAngina_Y), igual à seção 6.15
# - remainder="passthrough": FastingBS e Colesterol_ausente passam sem mudança
preparacao = ColumnTransformer([
    ("padronizar", StandardScaler(), continuas),
    ("one_hot", OneHotEncoder(drop="if_binary"), categoricas)],
    remainder="passthrough")

# Esteira completa: colesterol -> preparação -> modelo
esteira = Pipeline([
    ("colesterol", TratadorColesterol()),
    ("preparacao", preparacao),
    ("modelo", LogisticRegression(max_iter=1000, random_state=42))])

# Treina TODA a esteira de uma vez, só com o treino
esteira.fit(X_treino, y_treino)

# ---------- Conferência no teste ----------

# Probabilidade de doença para cada paciente de teste
probabilidades = esteira.predict_proba(X_teste)[:, 1]

# Aplica o limiar escolhido
previsoes = (probabilidades >= LIMIAR).astype(int)

# Mostra as métricas (devem bater com a seção 6.23)
print(f"=== Esteira completa no TESTE (limiar {LIMIAR:.2f}) ===")
print(f"Recall:   {recall_score(y_teste, previsoes) * 100:.1f}%")
print(f"Precisão: {precision_score(y_teste, previsoes) * 100:.1f}%")
print(f"Acurácia: {accuracy_score(y_teste, previsoes) * 100:.1f}%")
print()

# ---------- Salvando o modelo ----------

# Cria a pasta modelos se ela não existir
os.makedirs("modelos", exist_ok=True)

# Pacote salvo: a esteira treinada + informações para quem for usá-la
pacote = {
    "esteira": esteira,                              # modelo completo treinado
    "limiar": LIMIAR,                                # ponto de corte da decisão
    "colunas_entrada": list(X_treino.columns),       # campos que o paciente precisa ter
    "versao_sklearn": sklearn.__version__}           # versão usada no treino

# Grava o pacote no arquivo
joblib.dump(pacote, CAMINHO_MODELO)

# Mostra o tamanho do arquivo salvo, em KB
tamanho_kb = os.path.getsize(CAMINHO_MODELO) / 1024
print(f"Modelo salvo em {CAMINHO_MODELO} ({tamanho_kb:.1f} KB)")
print()

# ---------- Prova: carregar o arquivo e prever de novo ----------

# Carrega o pacote do disco, como a API vai fazer
carregado = joblib.load(CAMINHO_MODELO)

# Faz as previsões com a esteira CARREGADA do arquivo
prob_carregado = carregado["esteira"].predict_proba(X_teste)[:, 1]
prev_carregado = (prob_carregado >= carregado["limiar"]).astype(int)

# Confere se as previsões são idênticas às do modelo em memória
iguais = (prev_carregado == previsoes).all()

print("=== Conferência do arquivo salvo ===")
print(f"Limiar guardado: {carregado['limiar']}")
print(f"Campos de entrada: {len(carregado['colunas_entrada'])} -> {carregado['colunas_entrada']}")
print(f"Versão do scikit-learn: {carregado['versao_sklearn']}")
print(f"Previsões idênticas ao modelo em memória: {iguais}")
