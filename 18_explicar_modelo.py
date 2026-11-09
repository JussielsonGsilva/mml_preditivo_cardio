# Biblioteca padrão para lidar com pastas
import os

# NumPy para calcular a razão de chances (exponencial dos pesos)
import numpy as np

# Pandas para ler os dados e montar a tabela de pesos
import pandas as pd

# Matplotlib para desenhar o gráfico
import matplotlib

# Modo "Agg": salva direto em arquivo, sem abrir janela
matplotlib.use("Agg")

# Módulo de gráficos com o apelido "plt"
import matplotlib.pyplot as plt

# O modelo escolhido
from sklearn.linear_model import LogisticRegression

# Mostra todas as linhas e colunas da tabela, sem cortar
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)

# Lê o treino padronizado (o modelo final é treinado com ele)
treino = pd.read_csv("dados/treino_padronizado.csv")

# Separa entradas (X) e resposta certa (y)
X_treino = treino.drop(columns="HeartDisease")
y_treino = treino["HeartDisease"]

# Treina a Regressão Logística (mesma configuração de sempre)
modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(X_treino, y_treino)

# modelo.coef_[0] guarda o PESO aprendido para cada variável,
# na mesma ordem das colunas de X_treino
pesos = pd.Series(modelo.coef_[0], index=X_treino.columns)

# Razão de chances: exp(peso) diz por quanto a CHANCE de doença
# é multiplicada quando a variável sobe 1 unidade
# (nas colunas 0/1: quando a característica está presente;
#  nas contínuas: quando sobe 1 desvio padrão)
razao_chances = np.exp(pesos)

# Monta a tabela, ordenada do peso mais positivo para o mais negativo
tabela = pd.DataFrame({"Peso": pesos.round(3),
                       "Razão de chances": razao_chances.round(2)})
tabela = tabela.sort_values("Peso", ascending=False)

# Mostra a tabela
print("=== Pesos da Regressão Logística (19 variáveis) ===")
print("Peso > 0: aumenta o risco | Peso < 0: diminui o risco")
print()
print(tabela)
print()

# ---------- Gráfico ----------

# Garante que a pasta graficos existe
os.makedirs("graficos", exist_ok=True)

# Ordena do menor para o maior para o maior peso ficar no topo do gráfico
pesos_ordenados = pesos.sort_values()

# Vermelho para peso positivo (aumenta risco), verde para negativo
cores = ["#E53935" if valor > 0 else "#4CAF50" for valor in pesos_ordenados]

# Cria uma figura de 9 x 7 polegadas (19 barras precisam de altura)
plt.figure(figsize=(9, 7))

# Barras horizontais: nomes no eixo Y, pesos no eixo X
plt.barh(pesos_ordenados.index, pesos_ordenados.values, color=cores)

# Linha vertical preta no zero
plt.axvline(0, color="black", linewidth=0.8)

# Título e nome do eixo X
plt.title("Peso de cada variável na Regressão Logística")
plt.xlabel("Peso (vermelho = aumenta o risco | verde = diminui o risco)")

# Ajusta as margens e salva a imagem
plt.tight_layout()
plt.savefig("graficos/08_pesos_modelo.png")

# Avisa onde a imagem foi salva
print("Gráfico salvo em graficos/08_pesos_modelo.png")
