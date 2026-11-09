# Pandas para ler os dados e montar a tabela de resultados
import pandas as pd

# Os dois modelos que serão comparados
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Padronizador e ferramentas para montar a "esteira" (pipeline)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Divisor em 5 partes estratificadas e a função de validação cruzada
from sklearn.model_selection import StratifiedKFold, cross_validate

# Mostra todas as colunas da tabela final, sem cortar com "..."
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Lê SÓ O TREINO (o teste continua guardado para a avaliação final)
treino = pd.read_csv("dados/treino_final.csv")

# Separa entradas (X) e resposta certa (y)
X = treino.drop(columns="HeartDisease")
y = treino["HeartDisease"]

# Colunas contínuas que a Regressão Logística precisa padronizar
continuas = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

# ---------- Regressão Logística dentro de uma esteira (Pipeline) ----------

# Etapa 1 da esteira: padroniza só as colunas contínuas;
# remainder="passthrough" deixa as demais colunas passarem sem mudança
padronizacao = ColumnTransformer(
    [("padronizar", StandardScaler(), continuas)],
    remainder="passthrough")

# Esteira completa: primeiro padroniza, depois treina o modelo.
# Em cada rodada da validação cruzada, a padronização aprende
# média e desvio SÓ com as 4 partes de treino daquela rodada
logistica = Pipeline([
    ("padronizacao", padronizacao),
    ("modelo", LogisticRegression(max_iter=1000, random_state=42))])

# ---------- Random Forest (não precisa de padronização) ----------

floresta = RandomForestClassifier(n_estimators=300, random_state=42)

# ---------- Validação cruzada ----------

# Divide o treino em 5 partes (n_splits=5), mantendo a proporção
# de doentes em cada parte (estratificado); shuffle embaralha antes;
# random_state=42 deixa a divisão sempre igual
divisor = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lista para guardar o resumo de cada modelo
resumo = []

# Roda a validação cruzada para cada modelo
for nome, modelo in [("Regressão Logística", logistica),
                     ("Random Forest", floresta)]:

    # Treina e avalia 5 vezes; cada vez uma parte diferente vira "teste"
    resultado = cross_validate(modelo, X, y, cv=divisor,
                               scoring=["accuracy", "recall"])

    # Recall e acurácia de cada uma das 5 rodadas, em %
    recalls = resultado["test_recall"] * 100
    acuracias = resultado["test_accuracy"] * 100

    # Mostra o resultado rodada por rodada;
    # float() deixa o número limpo na tela (sem o "np.float64")
    print(f"=== {nome} ===")
    print("Recall por rodada:   ", [float(round(v, 1)) for v in recalls])
    print("Acurácia por rodada: ", [float(round(v, 1)) for v in acuracias])
    print()

    # Guarda média e desvio padrão (o quanto os resultados variam)
    resumo.append({"Modelo": nome,
                   "Recall médio (%)": round(recalls.mean(), 1),
                   "Recall desvio (±)": round(recalls.std(), 1),
                   "Acurácia média (%)": round(acuracias.mean(), 1),
                   "Acurácia desvio (±)": round(acuracias.std(), 1)})

# Mostra a tabela final de comparação
print("=== Resumo da validação cruzada (5 rodadas, só treino) ===")
print(pd.DataFrame(resumo).set_index("Modelo"))
