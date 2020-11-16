import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy import stats


def nonsupervised_acc(clusters, y_true):
    '''Convierte los clusters en etiquetas utilizando
       la moda sobre las etiquetas reales'''
    map_code = dict()
    for lab in np.unique(clusters):
        group = clusters[clusters == lab]
        true = y_true[clusters == lab]
        mode = stats.mode(true).mode[0]
        map_code[lab] = mode

    y_pred = np.array([map_code[l] for l in clusters])
    # Extraemos el numero de casos positivos
    # (para luego calcular el acc con el total de etiquetas)
    correctos = np.sum([y_true==y_pred])
    return correctos

def train_dbscan(X, y, epsilon, vecinos):
    '''Recibe una matriz de datos y devuelve una lista con clusters'''
    db = DBSCAN(eps=epsilon, min_samples=vecinos)
    db = db.fit(X)
    clusters = db.labels_
    # Extraemos los outliers

    clusters_validos = clusters[clusters != -1]
    X_validos = X[clusters != -1]
    y_validos = y[clusters != -1]
    return clusters_validos, X_validos, y_validos

# Preparacion de datos
digits = load_digits()
labels = digits['target']
inputs = digits.data # samples x ancho*alto

# Estandarizamos los datos
X_estandar = StandardScaler().fit_transform(inputs)

# Valores para epsilon
rango_epsilon = np.linspace(1, 30, 10)
# Valores para vecinos
rango_vecinos = np.arange(2, 10, 1)

silhouette_scores = []
contador = 0 # Solo para saber el modelo que estamos entrenando
total_params = len(rango_epsilon)*len(rango_vecinos)
for epsilon in rango_epsilon:
    for vecinos in rango_vecinos:
        print('[INFO] Entrenando Modelo {}/{}'.format(contador, total_params), end='\r')

        # Entrenamos el modelo
        clusters, X_no_outliers, y_validos = train_dbscan(X_estandar, labels, epsilon, vecinos)

        # Verificamos si existen elementos dentro de un cluster
        # Dependiendo de las condiciones podriamos generar solo outliers
        # Ademas debemos tener al menos una etiqueta
        if X_no_outliers.shape[0] != 0 and np.unique(clusters).shape[0]>1:
            correctos = nonsupervised_acc(clusters, y_validos)
            accuracy = correctos/len(labels)
            silhouette_scores.append({
                'silhouette': silhouette_score(X_no_outliers, clusters),
                'accuracy': accuracy,
                'epsilon': epsilon,
                'vecinos': vecinos
            })
        # Aumentamos el contador
        contador+=1

# Visualizamos los coef. encontrados
plt.figure()
sil_scores = [resultado['silhouette'] for resultado in silhouette_scores]
acc_scores = [resultado['accuracy'] for resultado in silhouette_scores]
ticks_name = [r'$\epsilon: ${:.2f} - V: {}'.format(resultado['epsilon'], resultado['vecinos']) \
          for resultado in silhouette_scores]
plt.plot(sil_scores, marker='s', label='silhouette')
plt.plot(acc_scores, marker='o', label='accuracy')
plt.xticks(range(len(silhouette_scores)), ticks_name, rotation=90)
plt.legend()
plt.title('Busqueda de Hiperparametros')
plt.tight_layout()
plt.show()

# Extraemos la mejor configuracion basado en el coef. de silhouette
best_conf_index = np.argmax([resultado['accuracy'] for resultado in silhouette_scores])
best_conf = silhouette_scores[best_conf_index]

print('\n')
print('ACC: {:.2f}'.format(best_conf['accuracy']))
print('Silhouette: {:.2f}'.format(best_conf['silhouette']))
