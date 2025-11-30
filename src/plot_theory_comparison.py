import json
import matplotlib.pyplot as plt
import numpy as np
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

def load_results(filename='benchmark_strong_scaling.json'):
    if not os.path.isabs(filename):
        filename = os.path.join(RESULTS_DIR, filename)
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def calculate_theoretical_time_normalized(m, n, d, p, t_seq, alpha=1e-5, beta=1e-8):
    """
    Modelo teorico normalizado:
    T_par(p) = T_compute(p) + T_comm(p)
    
    T_compute(p) = (m * n * d / p) * t_op
    donde t_op = t_seq / (m * n * d)
    
    T_comm(p) = alpha * log2(p) + beta * (n*d + m/p)
    """
    t_op = t_seq / (m * n * d)
    t_compute = (m * n * d / p) * t_op
    t_comm = alpha * np.log2(max(p, 1)) + beta * (n * d + m / p)
    
    return t_compute + t_comm


def normalize_theoretical_expression(results, t_seq, m, n, d):
    """
    Normaliza la expresion teorica ajustando alpha y beta
    para que se aproxime a los datos experimentales
    """
    processes = np.array([r['processes'] for r in results])
    measured_times = np.array([r['time_compute_max'] for r in results])
    
    t_op = t_seq / (m * n * d)
    
    def model(p, alpha, beta):
        t_compute = (m * n * d / p) * t_op
        t_comm = alpha * np.log2(max(p, 1)) + beta * (n * d + m / p)
        return t_compute + t_comm
    
    from scipy.optimize import curve_fit
    
    try:
        params, _ = curve_fit(
            lambda p, a, b: np.array([model(pi, a, b) for pi in p]),
            processes,
            measured_times,
            p0=[1e-5, 1e-8],
            bounds=([0, 0], [1e-3, 1e-6])
        )
        alpha_opt, beta_opt = params
    except:
        alpha_opt, beta_opt = 1e-5, 1e-8
    
    return alpha_opt, beta_opt


def plot_theory_vs_practice(data, output_dir='results'):
    """
    Grafica comparacion entre tiempos teoricos y medidos
    """
    results = data['results']
    t_seq = data['sequential_time']
    meta = data['metadata']
    m, n, d = meta['m'], meta['n'], meta['d']
    
    processes = np.array([r['processes'] for r in results])
    measured_compute = np.array([r['time_compute_max'] for r in results])
    measured_total = np.array([r['time_total'] for r in results])
    
    alpha, beta = normalize_theoretical_expression(results, t_seq, m, n, d)
    
    p_range = np.linspace(1, max(processes), 100)
    theoretical_times = [calculate_theoretical_time_normalized(m, n, d, p, t_seq, alpha, beta) 
                        for p in p_range]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(p_range, theoretical_times, '--', linewidth=2, 
            label='Modelo Teorico Normalizado', color='#E63946')
    ax1.plot(processes, measured_compute, 'o-', linewidth=2, markersize=10,
            label='Tiempo Medido (Computo)', color='#06A77D')
    ax1.plot(processes, measured_total, 's-', linewidth=2, markersize=8,
            label='Tiempo Medido (Total)', color='#457B9D', alpha=0.7)
    
    ax1.set_xlabel('Numero de Procesos (p)', fontsize=12)
    ax1.set_ylabel('Tiempo (segundos)', fontsize=12)
    ax1.set_title('Comparacion: Modelo Teorico vs Experimental', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(processes)
    
    residuals = []
    for i, p in enumerate(processes):
        t_theory = calculate_theoretical_time_normalized(m, n, d, p, t_seq, alpha, beta)
        t_measured = measured_compute[i]
        residuals.append(abs(t_theory - t_measured) / t_measured * 100)
    
    ax2.bar(processes, residuals, width=0.6, color='#F77F00', alpha=0.8, edgecolor='black')
    ax2.axhline(y=10, color='red', linestyle='--', linewidth=2, label='10% threshold')
    
    ax2.set_xlabel('Numero de Procesos (p)', fontsize=12)
    ax2.set_ylabel('Error Relativo (%)', fontsize=12)
    ax2.set_title('Error entre Modelo Teorico y Experimental', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(processes)
    
    for i, (p, err) in enumerate(zip(processes, residuals)):
        ax2.text(p, err + 0.5, f'{err:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/theory_vs_practice.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/theory_vs_practice.png")
    plt.close()


def plot_scalability_analysis(data, output_dir='results'):
    """
    Analiza escalabilidad fuerte y debil
    """
    results = data['results']
    t_seq = data['sequential_time']
    
    processes = np.array([r['processes'] for r in results])
    speedup = np.array([r['speedup'] for r in results])
    efficiency = np.array([r['efficiency'] for r in results])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(processes, speedup, 'o-', linewidth=2, markersize=10,
            label='Speedup Medido', color='#2E86AB')
    ax1.plot(processes, processes, '--', linewidth=2,
            label='Speedup Ideal (S=p)', color='#A23B72')
    
    speedup_amdahl = []
    f = 0.98
    for p in processes:
        s = 1 / ((1 - f) + f / p)
        speedup_amdahl.append(s)
    ax1.plot(processes, speedup_amdahl, '-.', linewidth=2,
            label=f'Ley de Amdahl (f={f})', color='#F77F00')
    
    ax1.set_xlabel('Numero de Procesos (p)', fontsize=12)
    ax1.set_ylabel('Speedup S(p)', fontsize=12)
    ax1.set_title('Analisis de Speedup', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(processes)
    
    ax2.plot(processes, efficiency * 100, 'o-', linewidth=2, markersize=10,
            label='Eficiencia Medida', color='#06A77D')
    ax2.axhline(y=100, color='#A23B72', linestyle='--', linewidth=2,
               label='Eficiencia Ideal (100%)')
    ax2.axhline(y=80, color='orange', linestyle=':', linewidth=2,
               label='Umbral Aceptable (80%)')
    
    ax2.set_xlabel('Numero de Procesos (p)', fontsize=12)
    ax2.set_ylabel('Eficiencia E(p) (%)', fontsize=12)
    ax2.set_title('Analisis de Eficiencia', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(processes)
    ax2.set_ylim([0, max(efficiency) * 110])
    
    for p, e in zip(processes, efficiency * 100):
        ax2.text(p, e + 2, f'{e:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scalability_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/scalability_analysis.png")
    plt.close()


def plot_optimal_processes(data, output_dir='results'):
    """
    Determina y visualiza la cantidad optima de procesos
    """
    results = data['results']
    t_seq = data['sequential_time']
    
    processes = np.array([r['processes'] for r in results])
    time_total = np.array([r['time_total'] for r in results])
    efficiency = np.array([r['efficiency'] for r in results])
    speedup = np.array([r['speedup'] for r in results])
    
    cost = time_total * processes
    
    cost_efficiency = speedup / cost
    
    optimal_idx = np.argmax(cost_efficiency)
    optimal_p = int(processes[optimal_idx])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1.plot(processes, time_total, 'o-', linewidth=2, markersize=10, color='#2E86AB')
    ax1.axvline(x=optimal_p, color='red', linestyle='--', linewidth=2,
               label=f'Optimo: p={optimal_p}')
    ax1.set_xlabel('Numero de Procesos (p)', fontsize=12)
    ax1.set_ylabel('Tiempo Total (s)', fontsize=12)
    ax1.set_title('Tiempo de Ejecucion vs Procesos', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(processes)
    
    ax2.plot(processes, efficiency * 100, 's-', linewidth=2, markersize=10, color='#06A77D')
    ax2.axvline(x=optimal_p, color='red', linestyle='--', linewidth=2)
    ax2.axhline(y=80, color='orange', linestyle=':', linewidth=2, label='80% threshold')
    ax2.set_xlabel('Numero de Procesos (p)', fontsize=12)
    ax2.set_ylabel('Eficiencia (%)', fontsize=12)
    ax2.set_title('Eficiencia vs Procesos', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(processes)
    
    ax3.plot(processes, cost, '^-', linewidth=2, markersize=10, color='#E63946')
    ax3.axvline(x=optimal_p, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Numero de Procesos (p)', fontsize=12)
    ax3.set_ylabel('Costo (p * T(p))', fontsize=12)
    ax3.set_title('Costo Computacional vs Procesos', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(processes)
    
    ax4.plot(processes, cost_efficiency, 'o-', linewidth=2, markersize=10, color='#F77F00')
    ax4.axvline(x=optimal_p, color='red', linestyle='--', linewidth=2,
               label=f'Maximo en p={optimal_p}')
    ax4.scatter([optimal_p], [cost_efficiency[optimal_idx]], 
               s=200, color='red', marker='*', zorder=5)
    ax4.set_xlabel('Numero de Procesos (p)', fontsize=12)
    ax4.set_ylabel('Eficiencia de Costo (S/Costo)', fontsize=12)
    ax4.set_title('Metrica de Optimalidad: Speedup/Costo', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(processes)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/optimal_processes_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/optimal_processes_analysis.png")
    plt.close()
    
    print(f"\nAnalisis de cantidad optima de procesos:")
    print(f"  Proceso optimo: p = {optimal_p}")
    print(f"  Tiempo en p_opt: {time_total[optimal_idx]:.4f} s")
    print(f"  Speedup en p_opt: {speedup[optimal_idx]:.2f}x")
    print(f"  Eficiencia en p_opt: {efficiency[optimal_idx]*100:.1f}%")

def generate_comprehensive_report(data, output_dir='results'):
    """
    Genera reporte completo con analisis teorico
    """
    results = data['results']
    t_seq = data['sequential_time']
    meta = data['metadata']
    m, n, d = meta['m'], meta['n'], meta['d']
    
    alpha, beta = normalize_theoretical_expression(results, t_seq, m, n, d)
    
    report = []
    report.append("="*80)
    report.append("REPORTE COMPLETO: ANALISIS TEORICO VS EXPERIMENTAL")
    report.append("KNN Paralelo con MPI")
    report.append("="*80)
    
    report.append(f"\n1. PARAMETROS DEL EXPERIMENTO")
    report.append(f"   Dataset: {meta['dataset']}")
    report.append(f"   Muestras entrenamiento (n): {n}")
    report.append(f"   Muestras test (m): {m}")
    report.append(f"   Features (d): {d}")
    report.append(f"   k-vecinos: {meta['k']}")
    report.append(f"   Tiempo secuencial base: {t_seq:.4f} s")
    
    report.append(f"\n2. MODELO TEORICO NORMALIZADO")
    report.append(f"   Complejidad secuencial: T_seq = O(m * n * d)")
    report.append(f"   Complejidad paralela: T_par(p) = T_compute(p) + T_comm(p)")
    report.append(f"   ")
    report.append(f"   T_compute(p) = (m * n * d / p) * t_op")
    report.append(f"   T_comm(p) = alpha * log2(p) + beta * (n*d + m/p)")
    report.append(f"   ")
    report.append(f"   Parametros normalizados:")
    report.append(f"     t_op = T_seq/(m*n*d) = {t_seq/(m*n*d):.2e} s/operacion")
    report.append(f"     alpha (sincronizacion) = {alpha:.2e} s")
    report.append(f"     beta (transferencia) = {beta:.2e} s/byte")
    
    report.append(f"\n3. RESULTADOS EXPERIMENTALES")
    report.append(f"   {'p':<6} {'T_medido':<12} {'T_teorico':<12} {'Error':<10} {'Speedup':<10} {'Efic(%)'}")
    report.append(f"   {'-'*70}")
    
    for r in results:
        p = int(r['processes'])
        t_measured = r['time_compute_max']
        t_theory = calculate_theoretical_time_normalized(m, n, d, p, t_seq, alpha, beta)
        error = abs(t_theory - t_measured) / t_measured * 100
        speedup = r['speedup']
        efficiency = r['efficiency'] * 100
        
        report.append(f"   {p:<6} {t_measured:<12.4f} {t_theory:<12.4f} {error:<10.2f} "
                     f"{speedup:<10.2f} {efficiency:.1f}")
    
    processes = np.array([r['processes'] for r in results])
    optimal_idx = np.argmax([r['speedup'] / (r['time_total'] * r['processes']) for r in results])
    optimal_p = int(processes[optimal_idx])
    
    report.append(f"\n4. ANALISIS DE ESCALABILIDAD")
    report.append(f"   Tipo: Strong Scaling (problema fijo, variar p)")
    report.append(f"   Speedup maximo alcanzado: {max([r['speedup'] for r in results]):.2f}x")
    report.append(f"   Eficiencia maxima: {max([r['efficiency'] for r in results])*100:.1f}%")
    report.append(f"   Cantidad optima de procesos: p = {optimal_p}")
    report.append(f"   ")
    report.append(f"   Criterio de optimalidad:")
    report.append(f"     - Maximizar Speedup/Costo")
    report.append(f"     - Mantener eficiencia > 80%")
    report.append(f"     - Minimizar tiempo total")
    
    report.append(f"\n5. VALIDACION DEL MODELO")
    errors = []
    for r in results:
        p = int(r['processes'])
        t_theory = calculate_theoretical_time_normalized(m, n, d, p, t_seq, alpha, beta)
        error = abs(t_theory - r['time_compute_max']) / r['time_compute_max'] * 100
        errors.append(error)
    
    report.append(f"   Error relativo promedio: {np.mean(errors):.2f}%")
    report.append(f"   Error relativo maximo: {np.max(errors):.2f}%")
    report.append(f"   Error relativo minimo: {np.min(errors):.2f}%")
    
    if np.mean(errors) < 15:
        report.append(f"   Conclusion: Modelo teorico se ajusta BIEN a los datos")
    elif np.mean(errors) < 30:
        report.append(f"   Conclusion: Modelo teorico se ajusta RAZONABLEMENTE")
    else:
        report.append(f"   Conclusion: Modelo teorico requiere refinamiento")
    
    report.append(f"\n6. OBSERVACIONES")
    for i, r in enumerate(results):
        p = int(r['processes'])
        comm_pct = (r['time_comm'] / r['time_total']) * 100
        if comm_pct > 50:
            report.append(f"   ALERTA: p={p} tiene {comm_pct:.1f}% de overhead de comunicacion")
        if r['efficiency'] < 0.5:
            report.append(f"   ALERTA: p={p} tiene eficiencia critica ({r['efficiency']*100:.1f}%)")
    
    report.append(f"\n" + "="*80)
    
    report_text = "\n".join(report)
    
    output_file = os.path.join(output_dir, 'comprehensive_report.txt')
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(f"\nReporte guardado en: {output_file}")


def main():
    print("GENERANDO ANALISIS COMPLETO - TEORIA VS PRACTICA")
    
    data = load_results()
    output_dir = RESULTS_DIR
    
    print(f"\nGenerando graficas y analisis en '{output_dir}/'...")
    
    plot_theory_vs_practice(data, output_dir)
    plot_scalability_analysis(data, output_dir)
    plot_optimal_processes(data, output_dir)
    generate_comprehensive_report(data, output_dir)
    
    print("\nAnalisis completo finalizado!")

if __name__ == "__main__":
    main()
