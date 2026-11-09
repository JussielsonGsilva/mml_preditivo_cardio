# Pandas para ler, alterar e salvar os dados
import pandas as pd

# Lê os conjuntos de treino e teste gerados na etapa anterior
treino = pd.read_csv("dados/treino.csv")
teste = pd.read_csv("dados/teste.csv")

# ---------- 1. Mediana calculada SÓ NO TREINO ----------

# Pega só os valores reais de colesterol do treino (maiores que zero)
# e calcula a mediana deles
mediana_treino = treino.loc[treino["Cholesterol"] > 0, "Cholesterol"].median()

print(f"Mediana do colesterol (só treino, só valores reais): {mediana_treino} mg/dl")
print()

# ---------- 2. Coluna indicadora + 3. Preenchimento ----------

# Repete o mesmo tratamento para os dois conjuntos
for nome, tabela in [("Treino", treino), ("Teste", teste)]:

    # Conta quantos zeros existem antes do tratamento
    zeros_antes = (tabela["Cholesterol"] == 0).sum()

    # Cria a coluna nova: (== 0) gera True/False;
    # .astype(int) transforma True em 1 e False em 0.
    # Precisa vir ANTES do preenchimento, senão os zeros somem
    tabela["Colesterol_ausente"] = (tabela["Cholesterol"] == 0).astype(int)

    # Nas linhas onde o colesterol é zero, troca o valor pela mediana do TREINO
    tabela.loc[tabela["Cholesterol"] == 0, "Cholesterol"] = mediana_treino

    # Conta quantos zeros sobraram (o esperado é nenhum)
    zeros_depois = (tabela["Cholesterol"] == 0).sum()

    # Mostra o resumo do tratamento deste conjunto
    print(f"=== {nome} ===")
    print(f"Zeros antes: {zeros_antes} | zeros depois: {zeros_depois}")
    print(f"Colesterol_ausente = 1 em {tabela['Colesterol_ausente'].sum()} pacientes")
    print(f"Formato (linhas, colunas): {tabela.shape}")
    print()

# ---------- Salva em arquivos NOVOS (os anteriores ficam intactos) ----------

treino.to_csv("dados/treino_tratado.csv", index=False)
teste.to_csv("dados/teste_tratado.csv", index=False)

print("Arquivos salvos: dados/treino_tratado.csv e dados/teste_tratado.csv")
