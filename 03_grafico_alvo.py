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

# Conta quantos pacientes há em cada classe (0 e 1), em ordem
contagem = df["HeartDisease"].value_counts().sort_index()

# Calcula a porcentagem de cada classe; normalize=True devolve
# a fração (ex: 0.447), e * 100 transforma em porcentagem
percentual = (df["HeartDisease"].value_counts(normalize=True)
              .sort_index() * 100).round(1)

# Mostra contagem e porcentagem de cada classe no terminal
print("=== Distribuição da variável alvo ===")
print(f"0 - Sem doença: {contagem[0]} pacientes ({percentual[0]}%)")
print(f"1 - Com doença: {contagem[1]} pacientes ({percentual[1]}%)")
print(f"Total: {contagem.sum()} pacientes")
print()

# Pega a porcentagem da classe MENOR (a minoritária)
minoritaria = percentual.min()

# Classifica o balanceamento conforme a tabela de referência do README
if minoritaria >= 40:
    situacao = "BALANCEADA - nenhuma técnica necessária"
elif minoritaria >= 20:
    situacao = "DESBALANCEAMENTO LEVE - usar divisão estratificada"
elif minoritaria >= 5:
    situacao = "DESBALANCEAMENTO MODERADO - aplicar técnica de balanceamento"
else:
    situacao = "DESBALANCEAMENTO SEVERO - combinar técnicas"

# Mostra o resultado da classificação
print(f"Classe minoritária: {minoritaria}%")
print(f"Situação da base: {situacao}")
print()

# Cria uma figura de 6 x 4 polegadas
plt.figure(figsize=(6, 4))

# Desenha as barras: verde para sem doença, vermelho para com doença
barras = plt.bar(["Sem doença (0)", "Com doença (1)"],
                 contagem.values,
                 color=["#4CAF50", "#E53935"])

# Monta os rótulos de cada barra no formato "410 (44.7%)"
rotulos = [f"{qtd} ({pct}%)" for qtd, pct in zip(contagem.values, percentual.values)]

# Escreve os rótulos em cima de cada barra
plt.bar_label(barras, labels=rotulos)

# Aumenta o topo do eixo Y em 10% para o rótulo não encostar na borda
plt.ylim(0, contagem.max() * 1.1)

# Título do gráfico
plt.title("Distribuição de pacientes por diagnóstico")

# Nome do eixo Y
plt.ylabel("Quantidade de pacientes")

# Ajusta as margens para nada ficar cortado
plt.tight_layout()

# Salva o gráfico, substituindo a versão anterior
plt.savefig("graficos/01_distribuicao_alvo.png")

# Avisa onde a imagem foi salva
print("Gráfico salvo em graficos/01_distribuicao_alvo.png")
