# Biblioteca padrão para lidar com pastas
import os

# Pandas para ler os dados
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

# Lê os conjuntos padronizados
treino = pd.read_csv("dados/treino_padronizado.csv")
teste = pd.read_csv("dados/teste_padronizado.csv")

# Separa entradas (X) e resposta certa (y)
X_treino = treino.drop(columns="HeartDisease")
y_treino = treino["HeartDisease"]
X_teste = teste.drop(columns="HeartDisease")
y_teste = teste["HeartDisease"]

# Treina a Regressão Logística (mesma configuração das etapas anteriores)
modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(X_treino, y_treino)

# Previsões para os pacientes de teste
previsoes = modelo.predict(X_teste)

# Monta a matriz de confusão 2x2:
# linhas = diagnóstico REAL, colunas = PREVISÃO do modelo
matriz = confusion_matrix(y_teste, previsoes)

# .ravel() "achata" a matriz na ordem: VN, FP, FN, VP
vn, fp, fn, vp = matriz.ravel()

# Mostra as quatro casas da matriz com o significado de cada uma
print("=== Matriz de confusão (teste: 184 pacientes) ===")
print(f"Verdadeiros Negativos (saudável e previsto saudável): {vn}")
print(f"Falsos Positivos      (saudável mas previsto doente):  {fp}")
print(f"Falsos Negativos      (doente mas previsto saudável):  {fn}")
print(f"Verdadeiros Positivos (doente e previsto doente):      {vp}")
print()

# Calcula as quatro métricas a partir das previsões
print("=== Métricas ===")
print(f"Acurácia:  {accuracy_score(y_teste, previsoes) * 100:.1f}%  (acertos / total)")
print(f"Recall:    {recall_score(y_teste, previsoes) * 100:.1f}%  (doentes encontrados / doentes reais)")
print(f"Precisão:  {precision_score(y_teste, previsoes) * 100:.1f}%  (doentes reais / previstos doentes)")
print(f"F1-score:  {f1_score(y_teste, previsoes) * 100:.1f}%  (equilíbrio entre recall e precisão)")
print()

# Garante que a pasta graficos existe
os.makedirs("graficos", exist_ok=True)

# Prepara o desenho da matriz com nomes legíveis nos eixos
desenho = ConfusionMatrixDisplay(confusion_matrix=matriz,
                                 display_labels=["Sem doença", "Com doença"])

# Desenha a matriz em tons de azul (cmap="Blues"), sem barra de cores
desenho.plot(cmap="Blues", colorbar=False)

# Título do gráfico
plt.title("Matriz de confusão — Regressão Logística (teste)")

# Nomes dos eixos em português
plt.xlabel("Previsão do modelo")
plt.ylabel("Diagnóstico real")

# Ajusta as margens para nada ficar cortado
plt.tight_layout()

# Salva a imagem na pasta graficos
plt.savefig("graficos/05_matriz_confusao.png")

# Avisa onde a imagem foi salva
print("Gráfico salvo em graficos/05_matriz_confusao.png")
