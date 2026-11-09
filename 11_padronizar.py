# Pandas para ler e salvar os dados
import pandas as pd

# Padronizador do scikit-learn (média 0, desvio padrão 1)
from sklearn.preprocessing import StandardScaler

# Lê os conjuntos finais (100% numéricos)
treino = pd.read_csv("dados/treino_final.csv")
teste = pd.read_csv("dados/teste_final.csv")

# Colunas numéricas contínuas que serão padronizadas
# (as colunas 0/1 ficam como estão)
continuas = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

# Mostra média e desvio do treino ANTES da padronização;
# + 0.0 evita aparecer "-0.0" na tela
print("=== Treino ANTES (média e desvio padrão) ===")
print(treino[continuas].agg(["mean", "std"]).round(2) + 0.0)
print()

# Cria o padronizador
padronizador = StandardScaler()

# fit: aprende a média e o desvio de cada coluna, SÓ COM O TREINO
padronizador.fit(treino[continuas])

# Mostra o que foi aprendido (será usado no treino e no teste)
print("=== Valores aprendidos no treino ===")
aprendido = pd.DataFrame({"média": padronizador.mean_,
                          "desvio": padronizador.scale_},
                         index=continuas)
print(aprendido.round(2))
print()

# transform: aplica (valor - média) / desvio no treino e no teste,
# substituindo as colunas contínuas pelos valores padronizados
treino[continuas] = padronizador.transform(treino[continuas])
teste[continuas] = padronizador.transform(teste[continuas])

# Treino DEPOIS: deve ficar média 0 e desvio 1
print("=== Treino DEPOIS (esperado: média 0 e desvio 1) ===")
print(treino[continuas].agg(["mean", "std"]).round(2) + 0.0)
print()

# Teste DEPOIS: fica PERTO de 0 e 1, mas não exato (usou os valores do treino)
print("=== Teste DEPOIS (perto de 0 e 1, mas não exato) ===")
print(teste[continuas].agg(["mean", "std"]).round(2) + 0.0)
print()

# Salva em arquivos NOVOS (os anteriores ficam intactos)
treino.to_csv("dados/treino_padronizado.csv", index=False)
teste.to_csv("dados/teste_padronizado.csv", index=False)

print("Arquivos salvos: dados/treino_padronizado.csv e dados/teste_padronizado.csv")
