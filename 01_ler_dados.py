# Importa o pandas com o apelido "pd" (padrão usado no mundo todo)
import pandas as pd

# Lê o arquivo CSV e guarda numa "tabela" chamada df (DataFrame)
df = pd.read_csv("dados/heart.csv")

# Mostra quantas linhas e colunas a tabela tem: (linhas, colunas)
print("Formato:", df.shape)

# Linha em branco só para separar a saída
print()

# Mostra as 5 primeiras linhas da tabela
print(df.head())

print()

# Mostra o nome de cada coluna, o tipo de dado e quantos valores não estão vazios
df.info()
