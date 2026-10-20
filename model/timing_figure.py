from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


N_BETA_VALUES = 20

MEDIA = ("Square-particle medium", "Circular-particle medium")

TIMING = {
    "Square-particle medium": {
        "smp_build": 222.2692,
        "smp_energy_eval": 0.01816368,
        "mcrt_energy_run": 69.4029,
    },
    "Circular-particle medium": {
        "smp_build": 263.7786,
        "smp_energy_eval": 0.01819801,
        "mcrt_energy_run": 70.8555,
    },
}


def main():
    labels = ["Square", "Circle"]
    x = np.arange(len(labels))
    width = 0.34

    smp_build = np.array([TIMING[m]["smp_build"] for m in MEDIA])
    smp_eval_total = np.array(
        [N_BETA_VALUES * TIMING[m]["smp_energy_eval"] for m in MEDIA]
    )
    smp_total = smp_build + smp_eval_total
    mcrt_total = np.array([N_BETA_VALUES * TIMING[m]["mcrt_energy_run"] for m in MEDIA])

    smp_eval = np.array([TIMING[m]["smp_energy_eval"] for m in MEDIA])
    mcrt_eval = np.array([TIMING[m]["mcrt_energy_run"] for m in MEDIA])

    sweep_speedup = mcrt_total / smp_total
    eval_speedup = mcrt_eval / smp_eval

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    fig.suptitle(
        "Runtime advantage of the Semi-Markov renewal model",
        fontsize=17,
        fontweight="bold",
    )

    ax = axes[0]
    ax.bar(
        x - width / 2,
        smp_build,
        width,
        color="#7b90ff",
        label="SMP build once",
    )
    ax.bar(
        x - width / 2,
        smp_eval_total,
        width,
        bottom=smp_build,
        color="#c7cffd",
        label=f"SMP {N_BETA_VALUES} beta evaluations",
    )
    ax.bar(
        x + width / 2,
        mcrt_total,
        width,
        color="#3f4a54",
        label=f"MCRT {N_BETA_VALUES} full runs",
    )

    for i, (smp, mcrt, speedup) in enumerate(zip(smp_total, mcrt_total, sweep_speedup)):
        ax.text(
            x[i] - width / 2,
            smp + 35,
            f"{smp:.0f} s",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        ax.text(
            x[i] + width / 2,
            mcrt + 35,
            f"{mcrt:.0f} s",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        ax.text(
            x[i],
            max(smp, mcrt) + 150,
            f"{speedup:.1f}x faster",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#2f3941",
        )

    ax.set_title(f"Total time for a {N_BETA_VALUES}-value beta sweep", fontsize=12, pad=10)
    ax.set_ylabel("Runtime (seconds)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, mcrt_total.max() * 1.25)
    ax.grid(axis="y", alpha=0.25)
    legend_handles, legend_labels = ax.get_legend_handles_labels()

    ax = axes[1]
    ax.bar(
        x - width / 2,
        smp_eval,
        width,
        color="#7b90ff",
        label="SMP additional beta",
    )
    ax.bar(
        x + width / 2,
        mcrt_eval,
        width,
        color="#3f4a54",
        label="MCRT additional beta",
    )

    for i, (smp, mcrt, speedup) in enumerate(zip(smp_eval, mcrt_eval, eval_speedup)):
        ax.text(
            x[i] - width / 2,
            smp * 1.7,
            f"{smp:.3f} s",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        ax.text(
            x[i] + width / 2,
            mcrt * 1.12,
            f"{mcrt:.1f} s",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        ax.text(
            x[i],
            210,
            f"{speedup:,.0f}x faster",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#2f3941",
        )

    ax.set_yscale("log")
    ax.set_ylim(0.006, 420)
    ax.set_title("One additional beta after SMP is built", fontsize=12, pad=10)
    ax.set_ylabel("Runtime (seconds, log scale)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", which="both", alpha=0.25)
    for ax in fig.axes:
        ax.grid(False, which="both", axis="both")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.08),
    )
    out_dir = Path.cwd() / "plots"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "timing_advantage_renewal_vs_mcrt.png"

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved {out_path}")

    plt.show()
    for label, sweep, additional in zip(labels, sweep_speedup, eval_speedup):
        print(
            f"{label}: {sweep:.2f}x faster for {N_BETA_VALUES} beta values; "
            f"{additional:.0f}x faster per additional beta after build."
        )


if __name__ == "__main__":
    main()
