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

# Lê a CÓPIA DE TRABALHO dos dados
df = pd.read_csv("dados/heart_trabalho.csv")

# Garante que a pasta graficos existe
os.makedirs("graficos", exist_ok=True)

# Separa a coluna MaxHR dos pacientes SEM doença (HeartDisease == 0)
sem_doenca = df[df["HeartDisease"] == 0]["MaxHR"]

# Separa a coluna MaxHR dos pacientes COM doença (HeartDisease == 1)
com_doenca = df[df["HeartDisease"] == 1]["MaxHR"]

# Mostra média e mediana de cada grupo no terminal
print("=== Frequência cardíaca máxima (bpm) por diagnóstico ===")
print(f"Sem doença: média {sem_doenca.mean():.1f} | mediana {sem_doenca.median():.1f}")
print(f"Com doença: média {com_doenca.mean():.1f} | mediana {com_doenca.median():.1f}")

# Calcula a diferença entre as médias dos dois grupos
diferenca = sem_doenca.mean() - com_doenca.mean()
print(f"Diferença entre as médias: {diferenca:.1f} bpm")
print()

# Cria uma figura de 8 x 5 polegadas
plt.figure(figsize=(8, 5))

# Histograma dos pacientes sem doença:
# bins=20 divide os valores em 20 faixas;
# alpha=0.6 deixa a cor semitransparente para ver a sobreposição
plt.hist(sem_doenca, bins=20, alpha=0.6, color="#4CAF50", label="Sem doença")

# Histograma dos pacientes com doença, por cima do anterior
plt.hist(com_doenca, bins=20, alpha=0.6, color="#E53935", label="Com doença")

# Linha vertical tracejada na média de cada grupo
plt.axvline(sem_doenca.mean(), color="#2E7D32", linestyle="--",
            label=f"Média sem doença ({sem_doenca.mean():.0f} bpm)")
plt.axvline(com_doenca.mean(), color="#B71C1C", linestyle="--",
            label=f"Média com doença ({com_doenca.mean():.0f} bpm)")

# Título do gráfico
plt.title("Frequência cardíaca máxima × diagnóstico")

# Nome do eixo X
plt.xlabel("Frequência cardíaca máxima (bpm)")

# Nome do eixo Y
plt.ylabel("Quantidade de pacientes")

# Mostra a legenda (cores e linhas)
plt.legend()

# Ajusta as margens para nada ficar cortado
plt.tight_layout()

# Salva a imagem na pasta graficos
plt.savefig("graficos/03_maxhr_vs_diagnostico.png")

# Avisa onde a imagem foi salva
print("Gráfico salvo em graficos/03_maxhr_vs_diagnostico.png")
