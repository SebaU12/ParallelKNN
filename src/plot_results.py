import json
import matplotlib.pyplot as plt
import numpy as np
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

def load_results(filename='benchmark_strong_scaling.json'):
    """Carga los resultados del benchmark desde results/ por defecto"""
    if not os.path.isabs(filename):
        filename = os.path.join(RESULTS_DIR, filename)

    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def plot_execution_times(data, output_dir='results'):
    """Grafica tiempos de ejecucion vs numero de procesos"""
    results = data['results']
    
    processes = [r['processes'] for r in results]
    time_total = [r['time_total'] for r in results]
    time_compute = [r['time_compute_max'] for r in results]
    time_comm = [r['time_comm'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(processes, time_total, 'o-', linewidth=2, markersize=8, label='Total Time')
    ax.plot(processes, time_compute, 's-', linewidth=2, markersize=8, label='Computation Time')
    ax.plot(processes, time_comm, '^-', linewidth=2, markersize=8, label='Communication Time')
    
    ax.set_xlabel('Number of Processes', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Execution Time vs Number of Processes', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(processes)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/time_vs_processes.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/time_vs_processes.png")
    plt.close()

def plot_speedup(data, output_dir='results'):
    """Grafica speedup vs numero de procesos"""
    results = data['results']
    
    processes = [r['processes'] for r in results]
    speedup_measured = [r['speedup'] for r in results]
    speedup_ideal = processes  # Speedup ideal = p
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(processes, speedup_measured, 'o-', linewidth=2, markersize=8, 
            label='Measured Speedup', color='#2E86AB')
    ax.plot(processes, speedup_ideal, '--', linewidth=2, 
            label='Ideal Speedup (S=p)', color='#A23B72')
    
    ax.set_xlabel('Number of Processes', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title('Speedup vs Number of Processes', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(processes)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speedup.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/speedup.png")
    plt.close()

def plot_efficiency(data, output_dir='results'):
    """Grafica eficiencia vs numero de procesos"""
    results = data['results']
    
    processes = [r['processes'] for r in results]
    efficiency = [r['efficiency'] * 100 for r in results]  # Convertir a porcentaje
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(processes, efficiency, 'o-', linewidth=2, markersize=8, 
            color='#F18F01', label='Measured Efficiency')
    ax.axhline(y=100, color='#A23B72', linestyle='--', linewidth=2, 
               label='Ideal Efficiency (100%)')
    
    ax.set_xlabel('Number of Processes', fontsize=12)
    ax.set_ylabel('Efficiency (%)', fontsize=12)
    ax.set_title('Parallel Efficiency vs Number of Processes', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(processes)
    ax.set_ylim([0, max(efficiency) * 1.1])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/efficiency.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/efficiency.png")
    plt.close()

def plot_flops(data, output_dir='results'):
    """Grafica GFLOPS vs numero de procesos"""
    results = data['results']
    
    processes = [r['processes'] for r in results]
    gflops = [r['gflops_per_sec'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(processes, gflops, width=0.6, color='#06A77D', alpha=0.8, edgecolor='black')
    
    for i, (p, g) in enumerate(zip(processes, gflops)):
        ax.text(p, g + 0.005, f'{g:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Number of Processes', fontsize=12)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax.set_title('Computational Performance vs Number of Processes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(processes)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/flops_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/flops_analysis.png")
    plt.close()

def plot_time_breakdown(data, output_dir='results'):
    """Grafica desglose de tiempos (stacked bar)"""
    results = data['results']
    
    processes = [r['processes'] for r in results]
    time_bcast = [r['time_bcast'] for r in results]
    time_scatter = [r['time_scatter'] for r in results]
    time_compute = [r['time_compute_max'] for r in results]
    time_gather = [r['time_gather'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    width = 0.6
    
    p1 = ax.bar(processes, time_bcast, width, label='Broadcast', color='#E63946')
    p2 = ax.bar(processes, time_scatter, width, bottom=time_bcast, 
                label='Scatter', color='#F77F00')
    
    bottom = [b + s for b, s in zip(time_bcast, time_scatter)]
    p3 = ax.bar(processes, time_compute, width, bottom=bottom, 
                label='Computation', color='#06A77D')
    
    bottom = [b + c for b, c in zip(bottom, time_compute)]
    p4 = ax.bar(processes, time_gather, width, bottom=bottom, 
                label='Gather', color='#457B9D')
    
    ax.set_xlabel('Number of Processes', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Time Breakdown by Phase', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xticks(processes)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/time_breakdown.png")
    plt.close()

def plot_communication_overhead(data, output_dir='results'):
    """Grafica porcentaje de tiempo en comunicacion vs computo"""
    results = data['results']
    
    processes = [r['processes'] for r in results]
    comm_pct = [(r['time_comm'] / r['time_total']) * 100 for r in results]
    comp_pct = [(r['time_compute_max'] / r['time_total']) * 100 for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(processes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, comp_pct, width, label='Computation', 
                   color='#06A77D', alpha=0.8)
    bars2 = ax.bar(x + width/2, comm_pct, width, label='Communication', 
                   color='#E63946', alpha=0.8)
    
    ax.set_xlabel('Number of Processes', fontsize=12)
    ax.set_ylabel('Percentage of Total Time (%)', fontsize=12)
    ax.set_title('Communication Overhead vs Computation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(processes)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/communication_overhead.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/communication_overhead.png")
    plt.close()

def generate_summary_table(data, output_dir='results'):
    """Genera una tabla resumen en formato texto"""
    results = data['results']
    metadata = data['metadata']
    
    summary = []
    summary.append("BENCHMARK SUMMARY - KNN PARALLEL WITH MPI")
    summary.append(f"\nDataset: {metadata['dataset']}")
    summary.append(f"Training samples: {metadata['train_size']}")
    summary.append(f"Test samples: {metadata['test_size']}")
    summary.append(f"Features: {metadata['features']}")
    summary.append(f"k-neighbors: {metadata['k']}")
    summary.append(f"Sequential baseline: {data['sequential_time']:.4f} s")
    summary.append(f"\n{'='*80}")
    summary.append(f"{'Proc':<6} {'Time(s)':<10} {'Speedup':<10} {'Effic(%)':<10} "
                  f"{'GFLOPS':<10} {'Comm%':<8}")
    summary.append("-"*80)
    
    for r in results:
        comm_pct = (r['time_comm'] / r['time_total']) * 100
        summary.append(f"{int(r['processes']):<6} {r['time_total']:<10.4f} "
                      f"{r['speedup']:<10.4f} {r['efficiency']*100:<10.2f} "
                      f"{r['gflops_per_sec']:<10.4f} {comm_pct:<8.2f}")
    
    summary.append("="*80)
    summary_text = "\n".join(summary)
    
    output_file = f'{output_dir}/summary.txt'
    with open(output_file, 'w') as f:
        f.write(summary_text)
    
    print(f"\nSaved: {output_file}")
    print("\n" + summary_text)

def main():
    print("GENERATING ANALYSIS PLOTS - KNN PARALLEL BETA 3")
    
    data = load_results()
    output_dir = RESULTS_DIR
    
    print(f"\nGenerating plots in '{output_dir}/' directory...")
    
    plot_execution_times(data, output_dir)
    plot_speedup(data, output_dir)
    plot_efficiency(data, output_dir)
    plot_flops(data, output_dir)
    plot_time_breakdown(data, output_dir)
    plot_communication_overhead(data, output_dir)
    generate_summary_table(data, output_dir)

if __name__ == "__main__":
    main()
