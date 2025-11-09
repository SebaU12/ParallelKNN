import subprocess
import json
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_experiment(num_processes, dataset_size='full', repetitions=3):
    """
    Ejecuta el experimento con un numero dado de procesos
    
    Args:
        num_processes: numero de procesos MPI
        dataset_size: 'full', 'half', 'double'
        repetitions: numero de repeticiones para promediar
    
    Returns:
        diccionario con resultados promediados
    """
    results = []
    
    for rep in range(repetitions):
        cmd = f"mpiexec -n {num_processes} python {os.path.join(SRC_DIR, 'knn_parallel_v3.py')} --quiet --size {dataset_size}"
        try:
            output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
            line = output.strip().split('\n')[-1]
            values = line.split(',')
            
            result = {
                'processes': int(values[0]),
                'accuracy': float(values[1]),
                'time_total': float(values[2]),
                'time_bcast': float(values[3]),
                'time_scatter': float(values[4]),
                'time_compute_max': float(values[5]),
                'time_compute_min': float(values[6]),
                'time_compute_avg': float(values[7]),
                'time_gather': float(values[8]),
                'time_comm': float(values[9]),
                'speedup': float(values[10]),
                'efficiency': float(values[11]),
                'gflops': float(values[12]),
                'gflops_per_sec': float(values[13]),
                'dataset_size': dataset_size
            }
            results.append(result)
            print(f"  Rep {rep+1}/{repetitions}: Time={result['time_total']:.4f}s, Speedup={result['speedup']:.2f}x")
        except subprocess.CalledProcessError as e:
            print(f"Error running with {num_processes} processes: {e}")
            return None
    
    avg_result = {key: np.mean([r[key] for r in results]) for key in results[0].keys() if key != 'dataset_size'}
    avg_result['dataset_size'] = dataset_size
    avg_result['std_time'] = np.std([r['time_total'] for r in results])
    avg_result['std_speedup'] = np.std([r['speedup'] for r in results])
    
    return avg_result


def calculate_theoretical_time(m, n, d, p, t_seq_base):
    """
    Calcula el tiempo teorico basado en la complejidad
    
    T_par = T_comm + T_compute
    T_comm = alpha * log(p) + beta * (n*d + m/p)
    T_compute = (m/p * n * d) * t_op
    
    Donde t_op es el tiempo por operacion individual
    """
    t_op = t_seq_base / (m * n * d)
    
    t_compute = (m / p) * n * d * t_op
    
    alpha = 1e-5
    beta = 1e-8
    t_comm = alpha * np.log2(p) + beta * (n * d + m / p)
    
    return t_compute + t_comm


def main():
    print("KNN PARALLEL BENCHMARK - EXTENDED ANALYSIS")
    print("=" * 60)
    
    mode = sys.argv[1] if len(sys.argv) > 1 else 'strong'
    
    if mode == 'strong':
        print("\nMODE: Strong Scaling (fixed problem size, varying processes)")
        processes_list = [1, 2, 4, 8]
        repetitions = 3
        dataset_sizes = ['full']
        
    elif mode == 'weak':
        print("\nMODE: Weak Scaling (m/p constant, varying both)")
        processes_list = [1, 2, 4, 8]
        repetitions = 3
        dataset_sizes = ['full', 'double', 'quad']
        
    elif mode == 'data':
        print("\nMODE: Data Scaling (varying dataset size, p=4)")
        processes_list = [4]
        repetitions = 3
        dataset_sizes = ['quarter', 'half', 'full', 'double']
    else:
        print(f"Unknown mode: {mode}")
        return
    
    print("\nRunning sequential baseline...")
    cmd = f"python {os.path.join(SRC_DIR, 'knn_sequential.py')}"
    try:
        output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
        for line in output.split('\n'):
            if 'Execution time' in line:
                t_seq = float(line.split(':')[1].strip().split()[0])
                print(f"Sequential time: {t_seq:.4f} s")
                break
    except:
        t_seq = 1.6091
        print(f"Using default sequential time: {t_seq:.4f} s")
    
    all_results = []
    
    for size in dataset_sizes:
        for p in processes_list:
            print(f"\nRunning with p={p}, size={size} ({repetitions} repetitions)...")
            result = run_experiment(p, size, repetitions)
            if result:
                all_results.append(result)
                print(f"  Avg time: {result['time_total']:.4f} +/- {result['std_time']:.4f} s")
                print(f"  Avg speedup: {result['speedup']:.4f} +/- {result['std_speedup']:.4f}x")
                print(f"  Efficiency: {100*result['efficiency']:.2f}%")
    
    m, n, d = 360, 1437, 64
    theoretical_times = []
    for r in all_results:
        p = int(r['processes'])
        t_theory = calculate_theoretical_time(m, n, d, p, t_seq)
        theoretical_times.append(t_theory)
    
    output_data = {
        'mode': mode,
        'sequential_time': t_seq,
        'results': all_results,
        'theoretical_times': theoretical_times,
        'metadata': {
            'repetitions': repetitions,
            'dataset': 'MNIST digits',
            'train_size': 1437,
            'test_size': 360,
            'features': 64,
            'k': 3,
            'm': m,
            'n': n,
            'd': d
        }
    }
    
    output_file = os.path.join(RESULTS_DIR, f'benchmark_{mode}_scaling.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    optimal_p = analyze_optimal_processes(all_results, t_seq)
    print(f"\nEstimated optimal number of processes: {optimal_p}")

def analyze_optimal_processes(results, t_seq):
    """
    Estima la cantidad optima de procesos basado en eficiencia
    Criterio: eficiencia > 80% y maximo speedup
    """
    best_p = 1
    best_metric = 0
    
    for r in results:
        p = int(r['processes'])
        eff = r['efficiency']
        speedup = r['speedup']
        
        if eff > 0.8:
            metric = speedup / p
            if metric > best_metric:
                best_metric = metric
                best_p = p
    
    return best_p

if __name__ == "__main__":
    main()
