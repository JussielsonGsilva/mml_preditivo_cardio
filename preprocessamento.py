# Módulo com o tratamento personalizado do colesterol.
# Fica num arquivo separado porque o modelo salvo depende desta classe:
# quem carregar o modelo (o script de treino E a API) precisa importá-la.

# Pandas para manipular as tabelas
import pandas as pd

# Classes base do scikit-learn para criar um "transformador" próprio,
# que funciona igual ao StandardScaler (com fit e transform)
from sklearn.base import BaseEstimator, TransformerMixin


class TratadorColesterol(BaseEstimator, TransformerMixin):
    """Trata o colesterol igual à seção 6.14 do README:
    cria Colesterol_ausente e troca os zeros pela mediana do treino."""

    # fit: APRENDE a mediana do colesterol (só valores reais, maiores que zero)
    def fit(self, X, y=None):

        # Guarda a mediana num atributo terminado em "_"
        # (padrão do scikit-learn para "valor aprendido no fit")
        self.mediana_ = X.loc[X["Cholesterol"] > 0, "Cholesterol"].median()

        # O scikit-learn exige que o fit devolva o próprio objeto
        return self

    # transform: APLICA o tratamento usando a mediana aprendida no fit
    def transform(self, X):

        # Trabalha numa cópia para nunca alterar a tabela original
        X = X.copy()

        # Cria a coluna indicadora ANTES de preencher (1 = era zero)
        X["Colesterol_ausente"] = (X["Cholesterol"] == 0).astype(int)

        # Troca os zeros pela mediana aprendida no treino
        X.loc[X["Cholesterol"] == 0, "Cholesterol"] = self.mediana_

        # Devolve a tabela tratada
        return X


# ---------- Teste rápido (só roda quando o arquivo é executado direto) ----------
if __name__ == "__main__":

    # Lê o treino no formato ORIGINAL (antes de qualquer tratamento)
    treino = pd.read_csv("dados/treino.csv")

    # Cria o tratador e aprende a mediana com o treino
    tratador = TratadorColesterol()
    tratador.fit(treino)

    # Aplica o tratamento
    tratado = tratador.transform(treino)

    # Mostra o resultado (deve bater com a seção 6.14 do README)
    print(f"Mediana aprendida: {tratador.mediana_} mg/dl")
    print(f"Zeros antes:  {(treino['Cholesterol'] == 0).sum()}")
    print(f"Zeros depois: {(tratado['Cholesterol'] == 0).sum()}")
    print(f"Colesterol_ausente = 1 em {tratado['Colesterol_ausente'].sum()} pacientes")
    print(f"Colunas: {tratado.shape[1]} (12 originais + Colesterol_ausente)")
