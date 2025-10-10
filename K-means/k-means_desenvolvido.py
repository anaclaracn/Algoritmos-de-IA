import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
import os
import time
import tracemalloc

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None, verbose=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.centroids = None
        self.labels_ = None

    def _inicializar_centroides(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices].astype(float)

    def _atribuir_clusters(self, X, centroids):
        distancias = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :])**2, axis=2)
        return np.argmin(distancias, axis=1)

    def _atualizar_centroides(self, X, labels):
        num_features = X.shape[1]
        novos = np.zeros((self.n_clusters, num_features))
        for i in range(self.n_clusters):
            pontos = X[labels == i]
            if pontos.shape[0] == 0:  # cluster vazio
                novos[i] = X[np.random.randint(0, X.shape[0])]
            else:
                novos[i] = pontos.mean(axis=0)
        return novos

    def fit(self, X):
        self.centroids = self._inicializar_centroides(X)
        for iteracao in range(self.max_iter):
            labels = self._atribuir_clusters(X, self.centroids)
            novos = self._atualizar_centroides(X, labels)
            deslocamento = np.linalg.norm(novos - self.centroids, axis=1).max()
            self.centroids = novos
            if self.verbose:
                print(f"Iter {iteracao:03d}: deslocamento máximo = {deslocamento:.6f}")
            if deslocamento <= self.tol:
                break
        self.labels_ = labels
        return self

    def predict(self, X):
        return self._atribuir_clusters(X, self.centroids)

    def inertia(self, X):
        return np.sum((X - self.centroids[self.labels_])**2)


def rodar_fluxo(k, X, y_true=None, seed=42, save_dir="resultados", verbose=False):
    os.makedirs(save_dir, exist_ok=True)
    melhor_solucao = None
    melhor_inercia = np.inf

    for init in range(10):  # várias inicializações
        modelo = KMeans(n_clusters=k, random_state=seed + init, verbose=verbose)
        modelo.fit(X)
        inertia = modelo.inertia(X)
        if inertia < melhor_inercia:
            melhor_inercia = inertia
            melhor_solucao = (modelo.labels_.copy(), modelo.centroids.copy(), seed + init)

    labels, centroids, used_seed = melhor_solucao
    sil = silhouette_score(X, labels) if k > 1 else float('nan')
    ari = adjusted_rand_score(y_true, labels) if y_true is not None else float('nan')
    nmi = normalized_mutual_info_score(y_true, labels) if y_true is not None else float('nan')

    # Reduz dimensões com PCA
    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X)
    centroids2 = pca.transform(centroids)

    plt.figure(figsize=(6, 5))
    for cluster in range(k):
        pts = X2[labels == cluster]
        plt.scatter(pts[:, 0], pts[:, 1], label=f'Cluster {cluster}', alpha=0.7, s=40)
    plt.scatter(centroids2[:, 0], centroids2[:, 1], marker='X', s=120, c='black', label='Centroids')
    plt.title(f'K-means Bruto (k={k}) - Silhouette: {sil:.4f}')
    plt.legend()
    filename = os.path.join(save_dir, f'kmeans_k{k}_seed{used_seed}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'k': k,
        'labels': labels,
        'centroids': centroids,
        'silhouette': sil,
        'inertia': melhor_inercia,
        'ARI': ari,
        'NMI': nmi,
        'seed_used': used_seed,
        'plot_file': filename
    }


def main():
    iris = load_iris()
    X = iris.data
    y_true = iris.target
    Xs = StandardScaler().fit_transform(X)

    resultados = []
    for k in [3, 5]:
        print(f"\nProcessando K-means Bruto para k={k} ...")
        tracemalloc.start()
        inicio = time.time()
        res = rodar_fluxo(k, Xs, y_true=y_true, seed=42, verbose=False)
        fim = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        resultados.append(res)
        print(f"  k = {k}, silhouette = {res['silhouette']:.4f}, inertia = {res['inertia']:.4f}")
        print(f"  ARI = {res['ARI']:.4f}, NMI = {res['NMI']:.4f}")
        print(f"  Figura salva em: {res['plot_file']}")
        print(f"  Tempo de execução: {fim - inicio:.4f} segundos")
        print(f"  Memória usada: atual {current/1024:.2f} KB, pico {peak/1024:.2f} KB")

    print("\nResumo comparativo:")
    for res in resultados:
        print(f"  k={res['k']} -> silhouette={res['silhouette']:.4f}, inertia={res['inertia']:.4f}, "
              f"ARI={res['ARI']:.4f}, NMI={res['NMI']:.4f}, seed={res['seed_used']}")

    print("\nComparação clusters x classes verdadeiras (apenas análise):")
    for res in resultados:
        df = pd.DataFrame({'true': y_true, 'cluster': res['labels']})
        cross = pd.crosstab(df['cluster'], df['true'])
        print(f"\nk = {res['k']}\n{cross}")


if __name__ == "__main__":
    main()
