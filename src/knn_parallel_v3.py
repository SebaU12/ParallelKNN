from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import sys

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

# Calcula FLOPs de region paralelizable: sqrt(sum((a-b)^2)
def calculate_flops(m_test, n_train, d):
    flops_per_distance = 3 * d
    total_distances = m_test * n_train
    return total_distances * flops_per_distance

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

k = 3
quiet_mode = '--quiet' in sys.argv

# Iniciar medicion de tiempo total
t_total_start = MPI.Wtime()

# Inicializacion en rank 0: carga de datos
if rank == 0:
    if not quiet_mode:
        print(f"KNN PARALLEL - BETA 3: Optimized with FLOPs Analysis")
        print(f"Number of processes: {size}")
        print(f"Loading MNIST digits dataset...")
    
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)
    
    m_test = X_test.shape[0]
    n_train = X_train.shape[0]
    d = X_train.shape[1]
    
    if not quiet_mode:
        print(f"Training samples: {n_train}")
        print(f"Test samples: {m_test}")
        print(f"Features: {d}")
        print(f"k = {k}")
        
        print(f"\nDistributing data among {size} processes...")
        X_test_chunks = np.array_split(X_test, size)
        for i in range(size):
            print(f"  Process {i}: {len(X_test_chunks[i])} test samples")
    else:
        X_test_chunks = np.array_split(X_test, size)
    
    y_test_chunks = np.array_split(y_test, size)
else:
    X_train = None
    y_train = None
    X_test_chunks = None
    y_test_chunks = None
    m_test = None
    n_train = None
    d = None

# Broadcast de dimensiones para calculo de FLOPs
dims = comm.bcast((m_test, n_train, d) if rank == 0 else None, root=0)
m_test, n_train, d = dims

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

if rank == 0 and not quiet_mode:
    print(f"\nStarting parallel computation...\n")

# Fase 3: Computo local
t_compute_start = MPI.Wtime()
local_predictions = [knn_predict(x, X_train, y_train, k) for x in local_X_test]
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

# Reducir tiempos de computo
t_compute_max = comm.reduce(t_compute_local, op=MPI.MAX, root=0)
t_compute_min = comm.reduce(t_compute_local, op=MPI.MIN, root=0)
t_compute_sum = comm.reduce(t_compute_local, op=MPI.SUM, root=0)

# Evaluacion y reporte en rank 0
if rank == 0:
    y_pred = []
    for sublist in all_predictions:
        y_pred.extend(sublist)
    
    y_true = []
    for sublist in all_y_test:
        y_true.extend(sublist)
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    accuracy = np.mean(y_pred == y_true)
    
    # Calcular metricas
    t_compute_avg = t_compute_sum / size
    t_comm = t_bcast + t_scatter + t_gather
    
    # Calcular FLOPs
    total_flops = calculate_flops(m_test, n_train, d)
    flops_per_process = calculate_flops(len(local_X_test), n_train, d)
    gflops = total_flops / 1e9
    gflops_per_sec = (total_flops / t_compute_max) / 1e9
    
    # Speedup y eficiencia
    t_sequential = 1.6091
    speedup = t_sequential / t_compute_max
    efficiency = speedup / size
    
    if quiet_mode:
        # Output formato CSV para benchmark
        print(f"{size},{accuracy:.6f},{t_total:.6f},{t_bcast:.6f},{t_scatter:.6f},"
              f"{t_compute_max:.6f},{t_compute_min:.6f},{t_compute_avg:.6f},"
              f"{t_gather:.6f},{t_comm:.6f},{speedup:.6f},{efficiency:.6f},"
              f"{gflops:.6f},{gflops_per_sec:.6f}")
    else:
        print(f"RESULTS - ACCURACY")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Total Samples: {len(y_true)}")
        print(f"Correct predictions: {np.sum(y_pred == y_true)}")
       
        print(f"Speedup: {speedup:.4f}, Efficiency: {efficiency:.4f}")
        print(f"Total FLOPs: {total_flops}, GFLOPs/s: {gflops_per_sec:.4f}")

        print()
        print("Sample predictions (first 10):")
        print("Predicted:", y_pred[:10])
        print("True:     ", y_true[:10])
        print()
