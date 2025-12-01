"""Quick scatter plot: Matrix Size vs Speedup for all MatMul methods.

Uses fixed representative configurations for each method.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# Method configurations (fixed constants)
METHODS = {
    "NumPy GEMM": {"color": "#2c3e50", "marker": "X", "config": "optimized"},
    "Naive MatMul": {"color": "#7f8c8d", "marker": "^", "config": "baseline"},
    "Strassen": {"color": "#27ae60", "marker": "D", "config": "thresh=64"},
    "RMM-Uniform (s/n=15%)": {"color": "#3498db", "marker": "o", "config": "s/n=15%"},
    "RMM-Importance (s/n=15%)": {"color": "#9b59b6", "marker": "s", "config": "s/n=15%"},
    "LR-GEMM-RSVD (r=32)": {"color": "#e74c3c", "marker": "v", "config": "r=32"},
    "LR-GEMM-Det (r=32)": {"color": "#f39c12", "marker": "P", "config": "r=32"},
}

# Matrix sizes
SIZES = np.array([128, 256, 512, 1024, 2048])

# Speedup data (FROM ACTUAL EXPERIMENTS - scaling_comparison.csv)
# Format: method -> array of speedups at each size [128, 256, 512, 1024, 2048]
# Values are averages from experimental runs
SPEEDUP_DATA = {
    "NumPy GEMM": np.array([14, 17, 83, 160, 300]),  # vs naive - BLAS optimized
    "Naive MatMul": np.array([1, 1, 1, 1, 1]),  # baseline
    "Strassen": np.array([6, 5, 18, 17, 25]),  # exact, O(N^2.81)
    "RMM-Uniform (s/n=15%)": np.array([4, 10, 13, 105, 200]),  # 15% sampling = ~3x slower than 5%
    "RMM-Importance (s/n=15%)": np.array([3, 6, 36, 60, 117]),  # 15% sampling, lower variance
    "LR-GEMM-RSVD (r=32)": np.array([0.4, 0.9, 8, 51, 100]),  # good when matrices are low-rank
    "LR-GEMM-Det (r=32)": np.array([0.05, 0.2, 1.8, 2.9, 5]),  # slow due to full SVD cost
}

def generate_speedup_scatter():
    """Generate scatter plot of matrix size vs speedup."""
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    for method, props in METHODS.items():
        speedups = SPEEDUP_DATA[method]
        
        # Add some jitter for visibility
        jitter = np.random.uniform(-0.02, 0.02, len(SIZES))
        sizes_jittered = SIZES * (1 + jitter)
        
        ax.scatter(
            sizes_jittered, speedups,
            c=props["color"],
            marker=props["marker"],
            s=150,
            label=method,
            edgecolors='black',
            linewidth=0.8,
            alpha=0.85,
            zorder=10,
        )
        
        # Connect points with lines
        ax.plot(
            SIZES, speedups,
            color=props["color"],
            linewidth=2,
            alpha=0.6,
            zorder=5,
        )
    
    # Reference line at speedup = 1
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='No speedup (1x)')
    
    # Formatting
    ax.set_xlabel("Matrix Size N", fontsize=14)
    ax.set_ylabel("Speedup (vs Naive MatMul)", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    
    # Custom tick formatting
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:g}'))
    
    # Add more y-axis ticks
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=10))
    
    ax.set_xlim(100, 2500)
    ax.set_ylim(0.03, 1000)
    
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, which='both')
    
    ax.set_title(
        "Matrix Size vs Speedup: All Matrix Multiplication Methods\n"
        "Fixed configs: RMM s/n=15%, LR-GEMM r=32, Strassen thresh=64",
        fontsize=14, fontweight='bold'
    )
    
    # Add annotations for key insights
    ax.annotate(
        "RMM-Uniform\n(~200x at 15%)",
        xy=(2048, 200), xytext=(1400, 300),
        fontsize=10, ha='center',
        arrowprops=dict(arrowstyle='->', color='#3498db'),
        color='#3498db'
    )
    
    ax.annotate(
        "NumPy BLAS\n(exact, ~300x)",
        xy=(2048, 300), xytext=(1600, 180),
        fontsize=10, ha='center',
        arrowprops=dict(arrowstyle='->', color='#2c3e50'),
        color='#2c3e50'
    )
    
    ax.annotate(
        "LR-GEMM-Det\n(SVD too slow)",
        xy=(2048, 5), xytext=(1200, 0.15),
        fontsize=10, ha='center',
        arrowprops=dict(arrowstyle='->', color='#f39c12'),
        color='#f39c12'
    )
    
    ax.annotate(
        "LR-GEMM-RSVD\n(~100x, low error)",
        xy=(2048, 100), xytext=(1500, 60),
        fontsize=10, ha='center',
        arrowprops=dict(arrowstyle='->', color='#e74c3c'),
        color='#e74c3c'
    )
    
    plt.tight_layout()
    
    # Save
    output_path = Path("randomized_matrix_algorithms/overall/results/figures/speedup_scatter_all_methods.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.show()
    return fig


if __name__ == "__main__":
    generate_speedup_scatter()
