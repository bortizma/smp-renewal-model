import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from core import RefractiveIndexConst
from gradient import reliable_gradient_field
from media import circle_medium_nonoverlap, square_medium_nonoverlap
from simulate import SimConfig, simulate_onesided_reflection
from smp_simulator import simulator_opaque

num_rays_smp = 20000
num_rays_mcrt = 5000
beta_values = np.linspace(0.0, 0.1, 20)


def make_medium(medium_kind, density, seed):
    if medium_kind == "square":
        return square_medium_nonoverlap(500, 20, density, seed=seed)

    if medium_kind == "circle":
        return circle_medium_nonoverlap(500, 500, radius=10, density=density, seed=seed)


def medium_title(medium_kind):
    return "Square-particle medium" if medium_kind == "square" else "Circular-particle medium"


def medium_suffix(medium_kind):
    return "square" if medium_kind == "square" else "circle"


def run_beta_comparison(medium_kind):
    med = make_medium(medium_kind, 0.3, seed=1)
    grad_field = reliable_gradient_field(med, smooth_sigma=2, normalize=True)

    cfg = SimConfig(
        medium=med,
        initial_direction=np.array([-1.0, 0.0]),
        initial_coord=np.array([med.shape[0] - 2, med.shape[1] // 2]),
        max_itr=2000,
        n_void=RefractiveIndexConst(1),
        n_matter=RefractiveIndexConst(3.5),
        wvlen=200e-9,
        pixel_length=0.01 * 200e-9,
        grad_field=grad_field,
        beta=0.0,
    )

    print(f"\n {medium_title(medium_kind)}")
    t0 = time.time()
    sim = simulator_opaque(cfg, num_direction_bins=8)
    sim.sim_opaque(
        num_sim=num_rays_smp,
        collect_chords=False,
        collect_ray_stats=True,
    )
    smp_build_time = time.time() - t0

    smp_reflectivity = []
    mcrt_reflectivity = []
    smp_eval_times = []
    mcrt_times = []

    for beta in beta_values:
        print(f"beta = {beta:.3f}")

        t0 = time.time()
        smp_value = sim.get_pathwise_energy_reflectivity(beta)
        smp_eval_times.append(time.time() - t0)
        smp_reflectivity.append(smp_value)

        t0 = time.time()
        vals, _ = simulate_onesided_reflection(
            med,
            beta=beta,
            num_sim=num_rays_mcrt,
        )
        mcrt_times.append(time.time() - t0)
        mcrt_reflectivity.append(np.mean(vals))

    smp_reflectivity = np.array(smp_reflectivity)
    mcrt_reflectivity = np.array(mcrt_reflectivity)

    plt.figure(figsize=(8, 5))

    plt.plot(
        beta_values,
        smp_reflectivity,
        "o-",
        color="#7b90ff",
        linewidth=2,
        label="Semi-Markov renewal",
    )

    plt.plot(
        beta_values,
        mcrt_reflectivity,
        "s--",
        color="#3f4a54",
        linewidth=2,
        label="MCRT",
    )

    plt.xlabel("Beta")
    plt.ylabel("Energy reflectivity")
    plt.title(f"Semi-Markov renewal reflectivity vs MCRT ({medium_title(medium_kind)})")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()

    out_path = f"smp_vs_mcrt_beta_{medium_suffix(medium_kind)}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved {out_path}")

    print("\n timing")
    print(f"smp build time: {smp_build_time:.4f} seconds")
    print(f"smp total evaluation time over all beta values: {sum(smp_eval_times):.8f} seconds")
    print(f"mcrt total time over all beta values: {sum(mcrt_times):.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--medium",
        choices=("square", "circle", "both"),
        default="both",
        help="Run square medium, circular medium, or both.",
    )
    args = parser.parse_args()

    medium_kinds = ("square", "circle") if args.medium == "both" else (args.medium,)
    for medium_kind in medium_kinds:
        run_beta_comparison(medium_kind)

    plt.show()
