# Pandas para ler e analisar os dados
import pandas as pd

# Lê a CÓPIA DE TRABALHO dos dados
df = pd.read_csv("dados/heart_trabalho.csv")

# ---------- Pergunta 1: em qual diagnóstico estão os zeros? ----------

# Filtra só as linhas onde o colesterol é zero
zeros = df[df["Cholesterol"] == 0]

# Conta quantos desses zeros são de cada diagnóstico (0 e 1)
zeros_por_classe = zeros["HeartDisease"].value_counts().sort_index()

# Calcula a porcentagem de cada diagnóstico entre os zeros
pct_zeros = (zeros["HeartDisease"].value_counts(normalize=True)
             .sort_index() * 100).round(1)

print("=== 1. Colesterol = 0 por diagnóstico ===")
print(f"Total de zeros: {len(zeros)}")
print(f"Sem doença (0): {zeros_por_classe[0]} ({pct_zeros[0]}%)")
print(f"Com doença (1): {zeros_por_classe[1]} ({pct_zeros[1]}%)")
print()

# ---------- Pergunta 2: correlação com e sem os zeros ----------

# Correlação usando TODOS os registros (com os zeros)
corr_com_zeros = df["Cholesterol"].corr(df["HeartDisease"])

# Filtra só as linhas com colesterol real (maior que zero)
sem_zeros = df[df["Cholesterol"] > 0]

# Correlação usando só os valores reais
corr_sem_zeros = sem_zeros["Cholesterol"].corr(sem_zeros["HeartDisease"])

print("=== 2. Correlação Cholesterol x HeartDisease ===")
print(f"Com os zeros: {corr_com_zeros:.3f}")
print(f"Sem os zeros: {corr_sem_zeros:.3f}")
print()

# ---------- Pergunta 3: média real do colesterol por grupo ----------

# Agrupa os valores reais por diagnóstico e calcula a média de cada grupo
media_real = sem_zeros.groupby("HeartDisease")["Cholesterol"].mean().round(1)

print("=== 3. Média do colesterol (só valores reais, em mg/dl) ===")
print(f"Sem doença (0): {media_real[0]}")
print(f"Com doença (1): {media_real[1]}")
