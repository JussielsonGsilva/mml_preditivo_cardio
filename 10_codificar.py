# Pandas para ler, codificar e salvar os dados
import pandas as pd

# Lê os conjuntos com o colesterol já tratado
treino = pd.read_csv("dados/treino_tratado.csv")
teste = pd.read_csv("dados/teste_tratado.csv")

# Colunas de texto com só 2 opções: viram UMA coluna 0/1
binarias = ["Sex", "ExerciseAngina"]

# Colunas de texto com 3 ou mais opções: viram uma coluna por categoria
multiplas = ["ChestPainType", "RestingECG", "ST_Slope"]


# Função que aplica a codificação em uma tabela (usada no treino e no teste)
def codificar(tabela):

    # get_dummies cria as colunas 0/1;
    # drop_first=True descarta a primeira categoria (ex: Sex_F),
    # deixando só uma coluna para as binárias (ex: Sex_M);
    # dtype=int grava 0 e 1 em vez de False e True
    tabela = pd.get_dummies(tabela, columns=binarias, drop_first=True, dtype=int)

    # Para as colunas com várias opções, mantém TODAS as categorias
    tabela = pd.get_dummies(tabela, columns=multiplas, dtype=int)

    # Devolve a tabela codificada
    return tabela


# Codifica o treino e o teste com a mesma função
treino_cod = codificar(treino)
teste_cod = codificar(teste)

# Garante que o teste tenha exatamente as mesmas colunas do treino,
# na mesma ordem; se faltar alguma, ela é criada com valor 0
teste_cod = teste_cod.reindex(columns=treino_cod.columns, fill_value=0)

# Mostra o formato de cada conjunto
print("=== Formato após codificação ===")
print(f"Treino: {treino_cod.shape}")
print(f"Teste:  {teste_cod.shape}")
print()

# Lista todas as colunas finais, numeradas
print("=== Colunas finais ===")
for numero, coluna in enumerate(treino_cod.columns, start=1):
    print(f"{numero:2}. {coluna}")
print()

# Conta quantas colunas ainda são texto (o esperado é zero)
texto_restante = treino_cod.select_dtypes(exclude="number").shape[1]
print(f"Colunas de texto restantes: {texto_restante}")

# Confere se treino e teste têm as mesmas colunas na mesma ordem
iguais = list(treino_cod.columns) == list(teste_cod.columns)
print(f"Treino e teste com as mesmas colunas: {iguais}")
print()

# Salva em arquivos NOVOS (os anteriores ficam intactos)
treino_cod.to_csv("dados/treino_final.csv", index=False)
teste_cod.to_csv("dados/teste_final.csv", index=False)

print("Arquivos salvos: dados/treino_final.csv e dados/teste_final.csv")
