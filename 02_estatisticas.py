# Importa o pandas com o apelido "pd"
import pandas as pd

# Faz o pandas mostrar todas as colunas, sem cortar com "..."
pd.set_option("display.max_columns", None)

# Aumenta a largura da saída para a tabela caber numa linha só
pd.set_option("display.width", 200)

# Lê o arquivo CSV para a tabela df
df = pd.read_csv("dados/heart.csv")

# Título da primeira parte da saída
print("=== Estatísticas das colunas numéricas ===")

# describe() calcula: count (quantidade), mean (média), std (desvio padrão),
# min (mínimo), 25%/50%/75% (quartis) e max (máximo).
# round(2) arredonda para 2 casas decimais, para facilitar a leitura
print(df.describe().round(2))

# Linha em branco para separar
print()

# Título da segunda parte da saída
print("=== Quantidade de valores iguais a zero ===")

# Percorre as duas colunas onde zero é impossível na vida real
for coluna in ["RestingBP", "Cholesterol"]:

    # (df[coluna] == 0) marca True onde o valor é zero;
    # .sum() conta quantos True existem
    zeros = (df[coluna] == 0).sum()

    # Mostra o nome da coluna e quantos zeros ela tem
    print(f"{coluna}: {zeros} registros com valor 0")
