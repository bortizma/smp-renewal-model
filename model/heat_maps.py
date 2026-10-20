import argparse

import matplotlib.pyplot as plt
import numpy as np

from core import RefractiveIndexConst
from gradient import reliable_gradient_field
from media import circle_medium_nonoverlap, square_medium_nonoverlap
from simulate import SimConfig, simulate_onesided_reflection
from smp_simulator import simulator_opaque
from validation import (
    calculate_theoretical_energy_reflectivity
)


def make_medium(medium_kind, density, seed):
    if medium_kind == "square":
        return square_medium_nonoverlap(500, 20, density, seed=seed)

    if medium_kind == "circle":
        return circle_medium_nonoverlap(500, 500, radius=10, density=density, seed=seed)


def medium_title(medium_kind):
    return "Square-particle medium" if medium_kind == "square" else "Circular-particle medium"


def medium_suffix(medium_kind):
    return "square" if medium_kind == "square" else "circle"


def renewal_data_for_medium(medium, num_rays, return_simulator=False):
    """Builds SMP/renewal data for one medium."""
    grad_field = reliable_gradient_field(medium, smooth_sigma=2, normalize=True)
    cfg = SimConfig(
        medium=medium,
        initial_direction=np.array([-1.0, 0.0]),
        initial_coord=np.array([medium.shape[0] - 2, medium.shape[1] // 2]),
        max_itr=2000,
        n_void=RefractiveIndexConst(1),
        n_matter=RefractiveIndexConst(3.5),
        wvlen=200e-9,
        pixel_length=0.01 * 200e-9,
        grad_field=grad_field,
        beta=0.0,
    )

    simulator = simulator_opaque(cfg, num_direction_bins=8)
    simulator.sim_opaque(
        num_sim=num_rays,
        collect_chords=True,
        collect_ray_stats=True,
    )

    P_matrix = simulator.get_transition_probabilities()
    erlang_data = simulator.fit_erlang_distributions()

    if return_simulator:
        return P_matrix, erlang_data, simulator
    return P_matrix, erlang_data


def print_accuracy_metrics(label, model_values, mcrt_values):
    residual = model_values - mcrt_values
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual**2))
    max_abs = np.max(np.abs(residual))
    bias = np.mean(residual)

    print(f"\n--- {label} accuracy metrics ---")
    print(f"MAE: {mae:.8e}")
    print(f"RMSE: {rmse:.8e}")
    print(f"Max absolute residual: {max_abs:.8e}")
    print(f"Bias/model - MCRT: {bias:.8e}")


def make_energy_heatmaps(
    grid_size=10,
    mcrt_rays_per_cell=5000,
    renewal_rays_per_porosity=20000,
    model="pathwise",
    medium_kind="square",
):
    porosity_vals = np.linspace(0.1, 0.5, grid_size)
    beta_vals = np.linspace(0.05, 0.2, grid_size)
    analytical_map = np.zeros((grid_size, grid_size))
    mcrt_map = np.zeros_like(analytical_map)

    for column, porosity in enumerate(porosity_vals):
        print(
            f"{medium_title(medium_kind)} | Porosity {porosity:.3f} "
            f"({column + 1}/{grid_size}): building SMP data"
        )
        medium = make_medium(medium_kind, porosity, seed=column + 1)

        P_matrix, erlang_data, simulator = renewal_data_for_medium(
            medium,
            renewal_rays_per_porosity,
            return_simulator=True,
        )

        for row, beta in enumerate(beta_vals):
            if model == "pathwise":
                analytical_map[row, column] = simulator.get_pathwise_energy_reflectivity(beta)
            elif model == "erlang":
                analytical_map[row, column] = calculate_theoretical_energy_reflectivity(
                    P_matrix,
                    erlang_data,
                    beta,
                )
            else:
                raise ValueError(f"Unknown model: {model}")

            mcrt_values, _ = simulate_onesided_reflection(
                medium,
                beta=beta,
                num_sim=mcrt_rays_per_cell,
            )
            mcrt_map[row, column] = mcrt_values.mean()
            print(
                f"  beta={beta:.3f}: model={analytical_map[row, column]:.5f}, "
                f"MCRT={mcrt_map[row, column]:.5f}"
            )

    residual_map = analytical_map - mcrt_map
    return porosity_vals, beta_vals, analytical_map, mcrt_map, residual_map


def plot_energy_heatmaps(
    porosity_vals,
    beta_vals,
    analytical_map,
    mcrt_map,
    residual_map,
    analytical_title="Semi-Markov renewal reflectivity",
    medium_kind="square",
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    reflectivity_max = max(analytical_map.max(), mcrt_map.max())
    residual_limit = np.abs(residual_map).max()
    if residual_limit == 0:
        residual_limit = 1e-12

    plots = (
        (analytical_map, analytical_title, "viridis", 0, reflectivity_max, "Reflectivity"),
        (mcrt_map, "MCRT ground-truth reflectivity", "viridis", 0, reflectivity_max, "Reflectivity"),
        (residual_map, "Residual: model - MCRT", "coolwarm", -residual_limit, residual_limit, "model - MCRT"),
    )

    for ax, (data, title, cmap, vmin, vmax, colorbar_label) in zip(axes, plots):
        image = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Solid-particle area fraction")
        ax.set_ylabel("Beta")
        ax.set_xticks(range(len(porosity_vals)), [f"{x:.2f}" for x in porosity_vals])
        ax.set_yticks(range(len(beta_vals)), [f"{x:.2f}" for x in beta_vals])
        fig.colorbar(image, ax=ax, label=colorbar_label)

    fig.suptitle(medium_title(medium_kind), fontsize=14)
    fig.savefig(
        f"reflectivity_heatmaps_{medium_suffix(medium_kind)}.png",
        dpi=300,
        bbox_inches="tight",
    )


def main():
    # to run a smoke test in case of changes
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a 2x2, 100-ray smoke test.")
    parser.add_argument(
        "--model",
        choices=("pathwise", "erlang"),
        default="pathwise",
        help="Use pathwise beta-zero ray lengths or the older direction-only Erlang renewal model.",
    )
    parser.add_argument("--skip-heatmap", action="store_true")
    parser.add_argument("--skip-probability", action="store_true")
    parser.add_argument(
        "--medium",
        choices=("square", "circle", "both"),
        default="both",
        help="Run square medium, circular medium, or both.",
    )
    args = parser.parse_args()

    if args.quick:
        grid_size = 2
        renewal_rays = 100
        mcrt_rays = 100
    else:
        grid_size = 10
        renewal_rays = 20000
        mcrt_rays = 5000

    medium_kinds = ("square", "circle") if args.medium == "both" else (args.medium,)

    for medium_kind in medium_kinds:
        print(f"\n===== {medium_title(medium_kind)} =====")

        if not args.skip_heatmap:
            energy_result = make_energy_heatmaps(
                grid_size=grid_size,
                mcrt_rays_per_cell=mcrt_rays,
                renewal_rays_per_porosity=renewal_rays,
                model=args.model,
                medium_kind=medium_kind,
            )
            _, _, analytical_map, mcrt_map, _ = energy_result
            print_accuracy_metrics(
                f"Energy reflectivity heatmap ({medium_title(medium_kind)})",
                analytical_map,
                mcrt_map,
            )

            title = (
                "Semi-Markov renewal reflectivity"
                if args.model == "pathwise"
                else "Direction-only Erlang renewal reflectivity"
            )
            plot_energy_heatmaps(
                *energy_result,
                analytical_title=title,
                medium_kind=medium_kind,
            )
    plt.show()


if __name__ == "__main__":
    main()
