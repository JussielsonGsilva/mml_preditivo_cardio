# Pandas para ler e salvar os dados
import pandas as pd

# Função do scikit-learn que faz a divisão treino/teste
from sklearn.model_selection import train_test_split

# Lê a CÓPIA DE TRABALHO dos dados
df = pd.read_csv("dados/heart_trabalho.csv")

# Mostra quantas linhas existem antes da limpeza
print(f"Linhas antes da limpeza: {len(df)}")

# Remove a linha com pressão arterial 0 (erro de cadastro):
# mantém só as linhas com RestingBP maior que zero
df = df[df["RestingBP"] > 0]

# Mostra quantas linhas sobraram
print(f"Linhas após remover RestingBP = 0: {len(df)}")
print()

# Divide a tabela inteira em treino (80%) e teste (20%):
# test_size=0.2      -> 20% vai para o teste
# stratify=...       -> mantém a mesma proporção de doentes nos dois conjuntos
# random_state=42    -> sorteio fixo, a divisão sai sempre igual
treino, teste = train_test_split(df,
                                 test_size=0.2,
                                 stratify=df["HeartDisease"],
                                 random_state=42)

# Mostra o tamanho de cada conjunto
print("=== Tamanho dos conjuntos ===")
print(f"Treino: {len(treino)} pacientes")
print(f"Teste:  {len(teste)} pacientes")
print()

# Confere se a proporção de doentes ficou parecida nos dois conjuntos
print("=== % com doença em cada conjunto ===")
print(f"Treino: {treino['HeartDisease'].mean() * 100:.1f}%")
print(f"Teste:  {teste['HeartDisease'].mean() * 100:.1f}%")
print()

# Salva cada conjunto num CSV próprio, dentro da pasta dados;
# index=False evita gravar a coluna de número da linha
treino.to_csv("dados/treino.csv", index=False)
teste.to_csv("dados/teste.csv", index=False)

# Avisa onde os arquivos foram salvos
print("Arquivos salvos: dados/treino.csv e dados/teste.csv")
