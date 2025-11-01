import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score


def rede_neural_perceptron(dataset_name, X, y, hidden_layers=(10, 10), max_iter=1000, random_state=42):
    print(f"\n----- Treinando Rede Neural para o dataset: {dataset_name} -----")

    # Normaliza os dados -> essencial para redes neurais
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split( X_scaled, y, test_size=0.3, random_state=random_state, stratify=y )

    # Cria o modelo MLP
    mlp = MLPClassifier( hidden_layer_sizes=hidden_layers, activation='relu', solver='adam', max_iter=max_iter, random_state=random_state )

    # Treina a rede
    mlp.fit(X_train, y_train)

    # Faz previsões
    y_pred = mlp.predict(X_test)

    # Avaliação
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')

    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Revocação: {rec:.4f}")
    print("\nRelatório de Classificação:\n")
    print(classification_report(y_test, y_pred))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.RdPu, cbar=False)
    plt.title(f"Matriz de Confusão - {dataset_name}")
    plt.xlabel("Previsto")
    plt.ylabel("Verdadeiro")
    plt.show()

    return {
        "dataset": dataset_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "report": classification_report(y_test, y_pred, output_dict=True)
    }


# IRIS
iris = load_iris()
result_iris = rede_neural_perceptron("Iris", iris.data, iris.target, hidden_layers=(8, 8))

# WINE
wine = load_wine()
result_wine = rede_neural_perceptron("Wine", wine.data, wine.target, hidden_layers=(12, 8, 4))

print("\nConclusão: ")
resumo = pd.DataFrame([result_iris, result_wine])[["dataset", "accuracy", "precision", "recall"]]
print(resumo)
