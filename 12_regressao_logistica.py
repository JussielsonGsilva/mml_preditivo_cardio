# Pandas para ler os dados
import pandas as pd

# O modelo de Regressão Logística do scikit-learn
from sklearn.linear_model import LogisticRegression

# Funções que calculam as métricas de acerto
from sklearn.metrics import accuracy_score, recall_score

# Lê os conjuntos padronizados
treino = pd.read_csv("dados/treino_padronizado.csv")
teste = pd.read_csv("dados/teste_padronizado.csv")

# X = variáveis de entrada (todas as colunas MENOS o alvo)
X_treino = treino.drop(columns="HeartDisease")
X_teste = teste.drop(columns="HeartDisease")

# y = resposta certa (só a coluna alvo)
y_treino = treino["HeartDisease"]
y_teste = teste["HeartDisease"]

# Mostra quantas variáveis de entrada o modelo vai usar
print(f"Variáveis de entrada: {X_treino.shape[1]}")
print()

# Cria o modelo:
# max_iter=1000   -> até 1000 tentativas para encontrar os melhores pesos
# random_state=42 -> resultado sempre igual (reprodutível)
modelo = LogisticRegression(max_iter=1000, random_state=42)

# Treina o modelo: ele aprende os pesos olhando X_treino e y_treino
modelo.fit(X_treino, y_treino)

# Faz as previsões para os pacientes de TESTE (que o modelo nunca viu)
previsoes = modelo.predict(X_teste)

# Acurácia: % de previsões certas, considerando todos os pacientes
acuracia = accuracy_score(y_teste, previsoes)

# Recall: % dos doentes de verdade que o modelo conseguiu identificar
recall = recall_score(y_teste, previsoes)

# Conta acertos totais e doentes encontrados, para mostrar em números absolutos
acertos = (previsoes == y_teste).sum()
doentes_reais = (y_teste == 1).sum()
doentes_encontrados = ((previsoes == 1) & (y_teste == 1)).sum()

# Mostra os resultados
print("=== Regressão Logística — resultado no TESTE ===")
print(f"Acurácia: {acuracia * 100:.1f}%  ({acertos} de {len(y_teste)} pacientes)")
print(f"Recall:   {recall * 100:.1f}%  ({doentes_encontrados} de {doentes_reais} doentes encontrados)")
print(f"Doentes NÃO detectados: {doentes_reais - doentes_encontrados}")
