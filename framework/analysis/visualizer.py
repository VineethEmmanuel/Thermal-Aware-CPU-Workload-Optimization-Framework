"""
Visualization and Terminal Formatting Utilities for DAA Analysis.

Provides:
- ASCIITableFormatter: Terminal-friendly formatted comparison tables.
- PlotGenerator: Matplotlib-based scientific visualization saving charts to disk.
"""

import os
from typing import Any, List, Optional, Sequence


class ASCIITableFormatter:
    """Formats 2D tabular data into clean, bordered terminal ASCII tables."""
    
    @staticmethod
    def render(headers: Sequence[str], rows: Sequence[Sequence[Any]], title: Optional[str] = None) -> str:
        """Renders ASCII table."""
        # Convert all cells to strings
        str_headers = [str(h) for h in headers]
        str_rows = [[str(cell) for cell in row] for row in rows]
        
        # Calculate maximum width per column
        col_widths = [len(h) for h in str_headers]
        for row in str_rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(cell))
                    
        # Total table width
        total_inner_width = sum(col_widths) + (3 * len(col_widths)) - 1
        
        lines: List[str] = []
        border = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        
        if title:
            lines.append("+" + "-" * (total_inner_width + 2) + "+")
            lines.append(f"| {title.center(total_inner_width)} |")
            
        lines.append(border)
        # Header line
        header_line = "| " + " | ".join(h.center(col_widths[i]) for i, h in enumerate(str_headers)) + " |"
        lines.append(header_line)
        lines.append(border.replace("-", "="))
        
        # Data rows
        for row in str_rows:
            # Pad row if incomplete
            padded = row + [""] * (len(col_widths) - len(row))
            row_line = "| " + " | ".join(padded[i].ljust(col_widths[i]) for i in range(len(col_widths))) + " |"
            lines.append(row_line)
            
        lines.append(border)
        return "\n".join(lines)


class PlotGenerator:
    """Generates benchmark plots and saves them as PNG images using matplotlib."""
    
    @staticmethod
    def generate_all_plots(benchmark_results: dict, output_dir: str = "benchmark_plots") -> List[str]:
        """Generates all 5 benchmark graphs."""
        try:
            import matplotlib
            matplotlib.use("Agg")  # Headless backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Warning] matplotlib is not installed. Skipping graphical plot generation.")
            return []

        os.makedirs(output_dir, exist_ok=True)
        saved_files: List[str] = []

        # 1. Knapsack Pseudo-Polynomial vs Polynomial Plot
        if "knapsack_scaling_w" in benchmark_results and "knapsack_scaling_n" in benchmark_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Subplot 1: Fixed n, Scale W exponentially
            w_data = benchmark_results["knapsack_scaling_w"]
            ws = [r.parameter_val for r in w_data]
            w_times = [r.elapsed_ms for r in w_data]
            ax1.plot(ws, w_times, 'ro-', linewidth=2, markersize=6)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_xlabel('Thermal Budget W (Capacity Units, Log Scale)', fontsize=11)
            ax1.set_ylabel('Execution Time (ms, Log Scale)', fontsize=11)
            ax1.set_title('Pseudo-Polynomial Growth: Scaling W (Fixed n=50)', fontsize=12, fontweight='bold')
            ax1.grid(True, which="both", ls="--", alpha=0.6)
            
            # Subplot 2: Fixed W, Scale n linearly
            n_data = benchmark_results["knapsack_scaling_n"]
            ns = [r.parameter_val for r in n_data]
            n_times = [r.elapsed_ms for r in n_data]
            ax2.plot(ns, n_times, 'bo-', linewidth=2, markersize=6)
            ax2.set_xlabel('Number of Tasks n (Linear Scale)', fontsize=11)
            ax2.set_ylabel('Execution Time (ms)', fontsize=11)
            ax2.set_title('Linear Growth in n: Scaling n (Fixed W=500)', fontsize=12, fontweight='bold')
            ax2.grid(True, ls="--", alpha=0.6)
            
            plt.tight_layout()
            p1 = os.path.join(output_dir, "knapsack_pseudo_polynomial_scaling.png")
            plt.savefig(p1, dpi=300)
            plt.close()
            saved_files.append(p1)

        # 2. Matrix Chain Multiplication DP vs Brute Force Plot
        if "mcm_dp_vs_bruteforce" in benchmark_results:
            fig, ax = plt.subplots(figsize=(8, 5))
            mcm_data = benchmark_results["mcm_dp_vs_bruteforce"]
            ns = [d["n"] for d in mcm_data]
            dp_times = [d["dp_ms"] for d in mcm_data]
            bf_times = [d["bf_ms"] for d in mcm_data]
            
            ax.plot(ns, dp_times, 'g^-', label='DP Tabulation O(n³)', linewidth=2)
            ax.plot(ns, bf_times, 'rs--', label='Brute Force Ω(4ⁿ / n^{3/2})', linewidth=2)
            ax.set_yscale('log')
            ax.set_xlabel('Number of Matrices in Pipeline (n)', fontsize=11)
            ax.set_ylabel('Execution Time (ms, Log Scale)', fontsize=11)
            ax.set_title('Matrix Chain Multiplication: DP O(n³) vs Catalan Explosion', fontsize=12, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, which="both", ls="--", alpha=0.6)
            
            plt.tight_layout()
            p2 = os.path.join(output_dir, "matrix_chain_dp_vs_bruteforce.png")
            plt.savefig(p2, dpi=300)
            plt.close()
            saved_files.append(p2)

        # 3. Binomial Heap vs Fibonacci Heap Operation Cost Plot
        if "heap_comparison" in benchmark_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            h_data = benchmark_results["heap_comparison"]
            ops_count = [d["num_ops"] for d in h_data]
            b_times = [d["b_ms"] for d in h_data]
            f_times = [d["f_ms"] for d in h_data]
            
            ax1.plot(ops_count, b_times, 'm-o', label='Binomial Heap O(log n)', linewidth=2)
            ax1.plot(ops_count, f_times, 'c-s', label='Fibonacci Heap O(1) Amortized', linewidth=2)
            ax1.set_xlabel('Total Heap Operations (Inserts + Decrease-Key)', fontsize=11)
            ax1.set_ylabel('Execution Time (ms)', fontsize=11)
            ax1.set_title('Decrease-Key Heavy Workload Runtime', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, ls="--", alpha=0.6)
            
            b_comps = [d["b_comparisons"] for d in h_data]
            f_comps = [d["f_comparisons"] for d in h_data]
            ax2.plot(ops_count, b_comps, 'm--', label='Binomial Heap Comparisons', linewidth=2)
            ax2.plot(ops_count, f_comps, 'c--', label='Fibonacci Heap Comparisons', linewidth=2)
            ax2.set_xlabel('Total Heap Operations', fontsize=11)
            ax2.set_ylabel('Comparisons Count', fontsize=11)
            ax2.set_title('Internal Comparisons Count', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, ls="--", alpha=0.6)
            
            plt.tight_layout()
            p3 = os.path.join(output_dir, "binomial_vs_fibonacci_heaps.png")
            plt.savefig(p3, dpi=300)
            plt.close()
            saved_files.append(p3)

        # 4. Telemetry Indexing Range Search Plot
        if "telemetry_indexing" in benchmark_results:
            fig, ax = plt.subplots(figsize=(8, 5))
            t_data = benchmark_results["telemetry_indexing"]
            sizes = [d["size"] for d in t_data]
            rbt_times = [d["rbt_ms"] for d in t_data]
            btree_times = [d["btree_ms"] for d in t_data]
            linear_times = [d["linear_ms"] for d in t_data]
            
            ax.plot(sizes, rbt_times, 'b-o', label='Red-Black Tree O(log N + k)', linewidth=2)
            ax.plot(sizes, btree_times, 'g-s', label='B-Tree (t=3) O(log_t N + k)', linewidth=2)
            ax.plot(sizes, linear_times, 'r--^', label='Linear Scan O(N)', linewidth=2)
            ax.set_xlabel('Telemetry Log Database Size (N records)', fontsize=11)
            ax.set_ylabel('Range Query Latency (ms)', fontsize=11)
            ax.set_title('Historical Telemetry Range Query Performance', fontsize=12, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, ls="--", alpha=0.6)
            
            plt.tight_layout()
            p4 = os.path.join(output_dir, "telemetry_indexing_range_query.png")
            plt.savefig(p4, dpi=300)
            plt.close()
            saved_files.append(p4)

        return saved_files
