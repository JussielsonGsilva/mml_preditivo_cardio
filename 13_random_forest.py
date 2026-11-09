# Pandas para ler os dados e montar a tabela de comparação
import pandas as pd

# Os dois modelos que serão comparados
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Funções que calculam as métricas
from sklearn.metrics import accuracy_score, recall_score

# ---------- Dados ----------

# Random Forest usa os dados SEM padronização (árvores não ligam para escala)
treino_rf = pd.read_csv("dados/treino_final.csv")
teste_rf = pd.read_csv("dados/teste_final.csv")

# Regressão Logística usa os dados PADRONIZADOS (como na etapa anterior)
treino_rl = pd.read_csv("dados/treino_padronizado.csv")
teste_rl = pd.read_csv("dados/teste_padronizado.csv")

# A resposta certa é a mesma nos dois arquivos de teste
y_treino = treino_rf["HeartDisease"]
y_teste = teste_rf["HeartDisease"]


# Função que treina um modelo, prevê no teste e devolve as métricas
def avaliar(modelo, treino, teste):

    # Separa as entradas (X) do alvo
    X_treino = treino.drop(columns="HeartDisease")
    X_teste = teste.drop(columns="HeartDisease")

    # Treina o modelo com os dados de treino
    modelo.fit(X_treino, y_treino)

    # Prevê os pacientes de teste
    previsoes = modelo.predict(X_teste)

    # Calcula as métricas
    acuracia = accuracy_score(y_teste, previsoes) * 100
    recall = recall_score(y_teste, previsoes) * 100

    # Conta os doentes que o modelo NÃO detectou
    nao_detectados = ((previsoes == 0) & (y_teste == 1)).sum()

    # Devolve os resultados num dicionário
    return {"Acurácia (%)": round(acuracia, 1),
            "Recall (%)": round(recall, 1),
            "Doentes não detectados": nao_detectados}


# ---------- Modelos ----------

# Random Forest:
# n_estimators=300 -> 300 árvores votando
# random_state=42  -> sorteios fixos, resultado reprodutível
floresta = RandomForestClassifier(n_estimators=300, random_state=42)

# Regressão Logística igual à etapa anterior (referência)
logistica = LogisticRegression(max_iter=1000, random_state=42)

# Avalia os dois modelos, cada um com seus dados
resultado_rl = avaliar(logistica, treino_rl, teste_rl)
resultado_rf = avaliar(floresta, treino_rf, teste_rf)

# ---------- Comparação ----------

# Monta uma tabela com um modelo por linha
comparacao = pd.DataFrame([resultado_rl, resultado_rf],
                          index=["Regressão Logística", "Random Forest"])

print("=== Comparação no TESTE (184 pacientes, 102 doentes) ===")
print(comparacao)
