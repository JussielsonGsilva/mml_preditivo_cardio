# Biblioteca padrão para lidar com pastas
import os

# Pandas para ler os dados e montar a tabela de comparação
import pandas as pd

# Matplotlib para desenhar o gráfico
import matplotlib

# Modo "Agg": salva direto em arquivo, sem abrir janela
matplotlib.use("Agg")

# Módulo de gráficos com o apelido "plt"
import matplotlib.pyplot as plt

# O modelo escolhido
from sklearn.linear_model import LogisticRegression

# Matriz de confusão, desenho da matriz e métricas
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, recall_score,
                             precision_score, f1_score)

# Mostra todas as colunas da tabela, sem cortar
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Limiar escolhido na seção 6.22 (validação cruzada no treino)
LIMIAR_ESCOLHIDO = 0.30

# Lê os conjuntos padronizados
treino = pd.read_csv("dados/treino_padronizado.csv")
teste = pd.read_csv("dados/teste_padronizado.csv")

# Separa entradas (X) e resposta certa (y)
X_treino = treino.drop(columns="HeartDisease")
y_treino = treino["HeartDisease"]
X_teste = teste.drop(columns="HeartDisease")
y_teste = teste["HeartDisease"]

# Treina a Regressão Logística com TODO o treino
modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(X_treino, y_treino)

# Probabilidade de "com doença" para cada paciente de teste
probabilidades = modelo.predict_proba(X_teste)[:, 1]

# Lista para guardar o resultado de cada limiar
linhas = []

# Avalia o limiar padrão e o limiar escolhido
for nome, limiar in [("Padrão (0.50)", 0.50),
                     (f"Escolhido ({LIMIAR_ESCOLHIDO:.2f})", LIMIAR_ESCOLHIDO)]:

    # Probabilidade >= limiar vira 1 (doente); abaixo vira 0 (saudável)
    previsoes = (probabilidades >= limiar).astype(int)

    # Guarda as métricas e a contagem de erros
    linhas.append({
        "Limiar": nome,
        "Recall (%)": round(recall_score(y_teste, previsoes) * 100, 1),
        "Precisão (%)": round(precision_score(y_teste, previsoes) * 100, 1),
        "F1 (%)": round(f1_score(y_teste, previsoes) * 100, 1),
        "Acurácia (%)": round(accuracy_score(y_teste, previsoes) * 100, 1),
        "Falsos negativos": int(((previsoes == 0) & (y_teste == 1)).sum()),
        "Falsos positivos": int(((previsoes == 1) & (y_teste == 0)).sum())})

# Monta e mostra a tabela de comparação
tabela = pd.DataFrame(linhas).set_index("Limiar")
print("=== AVALIAÇÃO FINAL no TESTE (184 pacientes, 102 doentes) ===")
print(tabela)
print()

# ---------- Matriz de confusão com o limiar escolhido ----------

# Previsões finais com o limiar escolhido
previsoes_finais = (probabilidades >= LIMIAR_ESCOLHIDO).astype(int)

# Monta a matriz 2x2 (linhas = real, colunas = previsão)
matriz = confusion_matrix(y_teste, previsoes_finais)

# Garante que a pasta graficos existe
os.makedirs("graficos", exist_ok=True)

# Desenha a matriz com nomes legíveis, em tons de verde
desenho = ConfusionMatrixDisplay(confusion_matrix=matriz,
                                 display_labels=["Sem doença", "Com doença"])
desenho.plot(cmap="Greens", colorbar=False)

# Título e nomes dos eixos
plt.title(f"Matriz de confusão final — limiar {LIMIAR_ESCOLHIDO:.2f} (teste)")
plt.xlabel("Previsão do modelo")
plt.ylabel("Diagnóstico real")

# Ajusta as margens e salva a imagem
plt.tight_layout()
plt.savefig("graficos/07_matriz_confusao_final.png")

# Avisa onde a imagem foi salva
print("Gráfico salvo em graficos/07_matriz_confusao_final.png")
