# Biblioteca padrão para lidar com pastas
import os

# NumPy para gerar a lista de limiares (0.20, 0.25, ... 0.60)
import numpy as np

# Pandas para ler os dados e montar a tabela
import pandas as pd

# Matplotlib para desenhar o gráfico
import matplotlib

# Modo "Agg": salva direto em arquivo, sem abrir janela
matplotlib.use("Agg")

# Módulo de gráficos com o apelido "plt"
import matplotlib.pyplot as plt

# Modelo, padronizador e esteira (igual à validação cruzada)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Divisor em 5 partes e função que gera previsões "fora da rodada"
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Métricas
from sklearn.metrics import recall_score, precision_score, accuracy_score

# Mostra todas as colunas da tabela, sem cortar
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Lê SÓ O TREINO (o teste continua guardado)
treino = pd.read_csv("dados/treino_final.csv")

# Separa entradas (X) e resposta certa (y)
X = treino.drop(columns="HeartDisease")
y = treino["HeartDisease"]

# Colunas contínuas a padronizar dentro da esteira
continuas = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

# Esteira: padroniza as contínuas e treina a Regressão Logística
esteira = Pipeline([
    ("padronizacao", ColumnTransformer(
        [("padronizar", StandardScaler(), continuas)],
        remainder="passthrough")),
    ("modelo", LogisticRegression(max_iter=1000, random_state=42))])

# Mesmo divisor da validação cruzada (5 partes, estratificado, fixo)
divisor = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# cross_val_predict: em cada rodada, prevê a parte que ficou de fora.
# method="predict_proba" devolve a PROBABILIDADE em vez de 0/1;
# [:, 1] pega a coluna da probabilidade de "com doença"
probabilidades = cross_val_predict(esteira, X, y, cv=divisor,
                                   method="predict_proba")[:, 1]

# Lista de limiares a testar: de 0.20 até 0.60, de 0.05 em 0.05
limiares = np.round(np.arange(0.20, 0.61, 0.05), 2)

# Lista para guardar o resultado de cada limiar
linhas = []

# Testa cada limiar
for limiar in limiares:

    # Probabilidade >= limiar vira 1 (doente); abaixo vira 0 (saudável)
    previsoes = (probabilidades >= limiar).astype(int)

    # Guarda as métricas e a contagem de cada tipo de erro
    linhas.append({
        "Limiar": limiar,
        "Recall (%)": round(recall_score(y, previsoes) * 100, 1),
        "Precisão (%)": round(precision_score(y, previsoes) * 100, 1),
        "Acurácia (%)": round(accuracy_score(y, previsoes) * 100, 1),
        "Falsos negativos": int(((previsoes == 0) & (y == 1)).sum()),
        "Falsos positivos": int(((previsoes == 1) & (y == 0)).sum())})

# Monta a tabela com um limiar por linha
tabela = pd.DataFrame(linhas).set_index("Limiar")

# Mostra a tabela (treino: 733 pacientes, 405 doentes)
print("=== Efeito do limiar (validação cruzada no TREINO: 733 pacientes, 405 doentes) ===")
print(tabela)
print()

# ---------- Gráfico ----------

# Garante que a pasta graficos existe
os.makedirs("graficos", exist_ok=True)

# Cria uma figura de 8 x 5 polegadas
plt.figure(figsize=(8, 5))

# Uma linha para cada métrica, com marcadores nos pontos
plt.plot(tabela.index, tabela["Recall (%)"], marker="o", color="#E53935", label="Recall")
plt.plot(tabela.index, tabela["Precisão (%)"], marker="o", color="#1E88E5", label="Precisão")
plt.plot(tabela.index, tabela["Acurácia (%)"], marker="o", color="#757575", label="Acurácia")

# Linha vertical tracejada no limiar padrão (0.50)
plt.axvline(0.50, color="black", linestyle="--", linewidth=0.8, label="Limiar padrão (0.50)")

# Título e nomes dos eixos
plt.title("Recall, precisão e acurácia conforme o limiar de decisão")
plt.xlabel("Limiar de decisão (probabilidade mínima para prever 'com doença')")
plt.ylabel("Percentual (%)")

# Mostra a legenda e uma grade leve para facilitar a leitura
plt.legend()
plt.grid(alpha=0.3)

# Ajusta as margens e salva a imagem
plt.tight_layout()
plt.savefig("graficos/06_limiar_decisao.png")

# Avisa onde a imagem foi salva
print("Gráfico salvo em graficos/06_limiar_decisao.png")
