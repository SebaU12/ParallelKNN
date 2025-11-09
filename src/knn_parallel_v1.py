from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import time

def euclidean_distance(a, b):
    """Calcula la distancia euclidiana entre dos puntos"""
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(test_point, X_train, y_train, k):
    """
    Predice la clase de un punto de test usando KNN
    
    Args:
        test_point: punto a clasificar
        X_train: conjunto de entrenamiento
        y_train: etiquetas de entrenamiento
        k: numero de vecinos a considerar
    
    Returns:
        clase predicha
    """
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

k = 3

# Inicializacion en rank 0: cargar y preparar datos
if rank == 0:
    print(f"KNN PARALLEL - BETA 1: Functional Implementation")
    print(f"Number of processes: {size}")
    print(f"Loading MNIST digits dataset...")
    
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"k = {k}")
    
    # Dividir X_test y y_test en chunks para scatter
    X_test_chunks = np.array_split(X_test, size)
    y_test_chunks = np.array_split(y_test, size)
    
    print(f"\nDistributing data among {size} processes...")
    for i in range(size):
        print(f"  Process {i}: {len(X_test_chunks[i])} test samples")
else:
    X_train = None
    y_train = None
    X_test_chunks = None
    y_test_chunks = None

# Fase 1: Broadcast (Todos los procesos necesitan X_train y y_train)
X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)

# Fase 2: Scatter (Distribuir X_test entre procesos)
local_X_test = comm.scatter(X_test_chunks, root=0)
local_y_test = comm.scatter(y_test_chunks, root=0)

if rank == 0:
    print(f"\nStarting parallel computation...")

# Fase 3: Computo local (Cada proceso predice sus puntos asignados)
start_compute = time.time()
local_predictions = []
for test_point in local_X_test:
    pred = knn_predict(test_point, X_train, y_train, k)
    local_predictions.append(pred)
end_compute = time.time()

# Fase 4: Gather (Recolectar todas las predicciones en rank 0)
all_predictions = comm.gather(local_predictions, root=0)
all_y_test = comm.gather(local_y_test, root=0)

# Evaluacion final en rank 0
if rank == 0:
    # Aplanar listas de listas
    y_pred = []
    for sublist in all_predictions:
        y_pred.extend(sublist)
    
    y_true = []
    for sublist in all_y_test:
        y_true.extend(sublist)
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    accuracy = np.mean(y_pred == y_true)
    
    print(f"RESULTS")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Correct predictions: {np.sum(y_pred == y_true)}/{len(y_true)}")
    
    # Verificacion: mostrar primeras 10 predicciones
    print("Sample predictions (first 10):")
    print("Predicted:", y_pred[:10])
    print("True:     ", y_true[:10])
    print()
