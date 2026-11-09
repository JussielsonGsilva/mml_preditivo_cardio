# Biblioteca padrão para lidar com pastas
import os

# Pandas para ler os dados e calcular a correlação
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

# Seleciona só as colunas numéricas (correlação não funciona com texto)
numericas = df.select_dtypes(include="number")

# .corr() calcula a correlação de todas as colunas entre si;
# ["HeartDisease"] pega só a correlação de cada uma com o alvo;
# .drop("HeartDisease") remove o alvo comparado com ele mesmo (sempre 1);
# .sort_values() ordena do menor para o maior
correlacao = (numericas.corr()["HeartDisease"]
              .drop("HeartDisease")
              .sort_values())

# Mostra os valores no terminal, do maior para o menor, com 3 casas
print("=== Correlação de cada variável com HeartDisease ===")
print(correlacao.sort_values(ascending=False).round(3))
print()

# Define a cor de cada barra: vermelho se positiva, verde se negativa
cores = ["#E53935" if valor > 0 else "#4CAF50" for valor in correlacao]

# Cria uma figura de 8 x 5 polegadas
plt.figure(figsize=(8, 5))

# Desenha barras HORIZONTAIS (barh): nomes no eixo Y, valores no eixo X
barras = plt.barh(correlacao.index, correlacao.values, color=cores)

# Escreve o valor de cada barra com 2 casas decimais
plt.bar_label(barras, labels=[f"{v:.2f}" for v in correlacao.values], padding=3)

# Linha vertical preta no zero, para separar positivo de negativo
plt.axvline(0, color="black", linewidth=0.8)

# Fixa o eixo X entre -1 e +1 (limites da correlação)
plt.xlim(-1, 1)

# Título do gráfico
plt.title("Correlação das variáveis numéricas com o diagnóstico")

# Nome do eixo X
plt.xlabel("Correlação (vermelho = aumenta o risco | verde = diminui o risco)")

# Ajusta as margens para nada ficar cortado
plt.tight_layout()

# Salva a imagem na pasta graficos
plt.savefig("graficos/04_correlacao_alvo.png")

# Avisa onde a imagem foi salva
print("Gráfico salvo em graficos/04_correlacao_alvo.png")
