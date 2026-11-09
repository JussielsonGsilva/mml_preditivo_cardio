# Biblioteca padrão para lidar com pastas
import os

# Pandas para ler e cruzar os dados
import pandas as pd

# Matplotlib para desenhar o gráfico
import matplotlib

# Modo "Agg": salva direto em arquivo, sem abrir janela
matplotlib.use("Agg")

# Módulo de gráficos com o apelido "plt"
import matplotlib.pyplot as plt

# Lê a CÓPIA DE TRABALHO dos dados
df = pd.read_csv("dados/heart_trabalho.csv")

# Garante que a pasta graficos existe
os.makedirs("graficos", exist_ok=True)

# Cruza tipo de dor (linhas) com diagnóstico (colunas 0 e 1)
# e conta quantos pacientes há em cada combinação
tabela = pd.crosstab(df["ChestPainType"], df["HeartDisease"])

# Renomeia as colunas 0 e 1 para nomes legíveis
tabela.columns = ["Sem doença", "Com doença"]

# Mostra a tabela de contagem no terminal
print("Pacientes por tipo de dor e diagnóstico:")
print(tabela)

# Linha em branco para separar
print()

# Calcula a porcentagem de doentes em cada tipo de dor:
# doentes do tipo / total do tipo * 100, arredondado em 1 casa
percentual = (tabela["Com doença"] / tabela.sum(axis=1) * 100).round(1)

# Mostra a porcentagem no terminal, do maior para o menor
print("% de pacientes COM doença em cada tipo de dor:")
print(percentual.sort_values(ascending=False))

# Desenha barras lado a lado (uma verde e uma vermelha por tipo de dor)
eixo = tabela.plot(kind="bar",
                   color=["#4CAF50", "#E53935"],
                   figsize=(8, 5))

# Escreve a quantidade em cima de cada barra (um grupo de cores por vez)
for grupo in eixo.containers:
    eixo.bar_label(grupo)

# Título do gráfico
plt.title("Tipo de dor no peito × diagnóstico")

# Nome do eixo X
plt.xlabel("Tipo de dor no peito (ASY = sem dor)")

# Nome do eixo Y
plt.ylabel("Quantidade de pacientes")

# Deixa os rótulos do eixo X na horizontal (sem girar)
plt.xticks(rotation=0)

# Título da legenda
plt.legend(title="Diagnóstico")

# Ajusta as margens para nada ficar cortado
plt.tight_layout()

# Salva a imagem na pasta graficos
plt.savefig("graficos/02_dor_peito_vs_diagnostico.png")

# Avisa onde a imagem foi salva
print()
print("Gráfico salvo em graficos/02_dor_peito_vs_diagnostico.png")
