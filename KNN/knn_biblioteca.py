import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

ks = [1, 3, 5, 7]

for k in ks:
    print(f"\nResultados com sklearn KNN para k={k}")
    
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    end = time.time()
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    
    print(f"Acuracia: {acc:.4f}")
    print(f"Precisa: {prec:.4f}")
    print(f"Revocacao: {rec:.4f}")
    print(f"Tempo Sklearn KNN: {end - start:.4f}s")
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot(cmap=plt.cm.RdPu)
    plt.title(f"Matriz de Confusão (Sklearn KNN) - k={k}")
    plt.show()


