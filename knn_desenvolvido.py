import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score
import time

def dist_euclidiana(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # dados de treino
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # guardar as previsões
        previsoes = []

        for amostra in X:
            rotulo_previsto = self.predict_individual(amostra)
            previsoes.append(rotulo_previsto)

        return np.array(previsoes)

    def predict_individual(self, x):
        # distância entre a amostra x e todos os pontos de treino
        distancias = []
        for ponto in self.X_train:
            d = dist_euclidiana(x, ponto)
            distancias.append(d)

        # ordena as distâncias e pega os índices dos k mais próximos
        indices_k_vizinhos = np.argsort(distancias)[:self.k]

        # pegar os rótulos desses vizinhos
        rotulos_vizinhos = []
        for i in indices_k_vizinhos:
            rotulos_vizinhos.append(self.y_train[i])

        # contagem do rótulo que mais aparece
        contagem = Counter(rotulos_vizinhos)
        rotulo_mais_comum = contagem.most_common(1)[0][0]

        return rotulo_mais_comum


data = load_iris()
X, y = data.data, data.target

# dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ks = [1, 3, 5, 7]

for k in ks:
    print(f"\nResultados para k={k} para o Conjunto de dados flor Iris")
    knn = KNN(k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)  # classes previstas das flores

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')

    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Revocação: {rec:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
    disp.plot(cmap=plt.cm.RdPu) 
    plt.title(f"Matriz de confusão - k={k}")
    plt.show()
        
    start = time.time()
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    end = time.time()
    print(f"Tempo KNN manual: {end - start:.4f}s")

