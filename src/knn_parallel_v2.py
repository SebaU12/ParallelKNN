from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

# Distancia euclidiana entre dos puntos
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Predicción de clase de un punto usando KNN
def knn_predict(test_point, X_train, y_train, k):
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

k = 3

# Iniciar medicion de tiempo total
t_total_start = MPI.Wtime()

# Inicializacion en rank 0: cargar y preparar datos
if rank == 0:
    print(f"KNN PARALLEL - BETA 2: Performance Analysis")
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

# Fase 1: Broadcast
t_bcast_start = MPI.Wtime()
X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)
t_bcast_end = MPI.Wtime()
t_bcast = t_bcast_end - t_bcast_start

# Fase 2: Scatter
t_scatter_start = MPI.Wtime()
local_X_test = comm.scatter(X_test_chunks, root=0)
local_y_test = comm.scatter(y_test_chunks, root=0)
t_scatter_end = MPI.Wtime()
t_scatter = t_scatter_end - t_scatter_start

if rank == 0:
    print(f"\nStarting parallel computation...")

# Fase 3: Computo local
t_compute_start = MPI.Wtime()
local_predictions = []
for test_point in local_X_test:
    pred = knn_predict(test_point, X_train, y_train, k)
    local_predictions.append(pred)
t_compute_end = MPI.Wtime()
t_compute_local = t_compute_end - t_compute_start

# Fase 4: Gather
t_gather_start = MPI.Wtime()
all_predictions = comm.gather(local_predictions, root=0)
all_y_test = comm.gather(local_y_test, root=0)
t_gather_end = MPI.Wtime()
t_gather = t_gather_end - t_gather_start

# Fin de medicion de tiempo total
t_total_end = MPI.Wtime()
t_total = t_total_end - t_total_start

# Reducir tiempos de computo para analisis
t_compute_max = comm.reduce(t_compute_local, op=MPI.MAX, root=0)
t_compute_min = comm.reduce(t_compute_local, op=MPI.MIN, root=0)
t_compute_sum = comm.reduce(t_compute_local, op=MPI.SUM, root=0)

# Evaluacion y reporte final en rank 0
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
    
    # Calcular metricas de comunicacion y computo
    t_compute_avg = t_compute_sum / size
    t_comm = t_bcast + t_scatter + t_gather
    
    # Calcular speedup y eficiencia 
    t_sequential = 1.6091  # Tiempo del secuencial reportado
    speedup = t_sequential / t_compute_max
    efficiency = speedup / size
    
    print(f"RESULTS - ACCURACY")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Correct predictions: {np.sum(y_pred == y_true)}/{len(y_true)}")
    
    # Verificacion: mostrar primeras 10 predicciones
    print("Sample predictions (first 10):")
    print("Predicted:", y_pred[:10])
    print("True:     ", y_true[:10])
    print()
