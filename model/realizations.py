#2 geometries x 5 area fractions x 5 medium realizations

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import core
from gradient import reliable_gradient_field
from media import circle_medium, square_medium

CODE_DIR = Path(r"C:\Users\Brisia\Dropbox\ECOlab-BrisiaOrtiz Martinez\ECOLAB_codes\ENFHT 2027")
OUTDIR = Path("csvs/realizations_by_area_fraction")
sys.path.insert(0, str(CODE_DIR))
single_ray_opaque_simulate = core.single_ray_opaque_simulate

media = ("square", "circle")
densities = [0.10, 0.20, 0.30, 0.40, 0.50]
medium_seeds = [1, 2, 3, 4, 5]
grid_size = 500
beta = 0.1
smp_rays = 20000
mcrt_rays = 5000
smooth_sigma = 2.0
max_irt = 2000
power_tolerance = 1e-6

initial_direction = np.array([-1.0, 0.0])

def make_medium(kind:str, density:float, seed:int) -> np.ndarray:
    if kind == "square":
        return square_medium(grid_size, 20, density, seed=seed)
    if kind == "circle":
        return circle_medium(grid_size, grid_size, radius=10, density=density, seed=seed)
    raise ValueError(f"unknown medium kind: {kind}")

def bottom_starts(med: np.ndarray) -> tuple[int, np.ndarray]:
    # middle bottom region
    start_row = med.shape[0]-2
    cand = np.arange(200, med.shape[1]-200)
    starts = cand[med[start_row, cand] == 0]

    # full bottom row
    if len(starts) == 0:
        cand = np.arange(med.shape[1])
        starts = cand[med[start_row, cand] == 0]
    if len(starts) == 0:
        raise ValueError("no void cells available along the bottom start row.")

    return start_row, starts


def build_smp_path_database(
    med: np.ndarray,
    grad_field: np.ndarray,
    num_rays: int,
    seed: int,
) -> tuple[list[dict], int, float]:
    rng = np.random.RandomState(seed)
    start_row, start_cols = bottom_starts(med)
    ray_stats = []
    failures = 0
    start_time = time.perf_counter()

    for _ in range(num_rays):
        initial_coord = np.array([start_row, rng.choice(start_cols)])
        try:
            single_ray_opaque_simulate(
                med,
                initial_direction,
                initial_coord=initial_coord,
                max_itr=max_irt,
                beta=0.0,
                grad_field=grad_field,
                power_tolerance=power_tolerance,
                ray_stats=ray_stats,
            )
        except AssertionError:
            failures += 1
            ray_stats.append({
                "exit_col": core.EXIT_LOST,
                "path_length": 0.0,
                "num_strikes": 0,
            })
    return ray_stats, failures, time.perf_counter() - start_time

def smp_outputs(ray_stats: list[dict], beta: float) -> tuple[float, float]:
    reflected = [
        stat for stat in ray_stats
        if stat["exit_col"] == core.EXIT_REFLECT
    ]

    probability = len(reflected) / len(ray_stats)
    energy = sum(
        math.exp(-2.0 * beta * stat["path_length"])
        for stat in reflected
    ) / len(ray_stats)

    return probability, energy

def run_mcrt(
    med: np.ndarray,
    grad_field: np.ndarray,
    beta: float,
    num_rays: int,
    seed: int,
) -> tuple[np.ndarray, int, float]:
    rng = np.random.RandomState(seed)
    start_row, start_cols = bottom_starts(med)
    values = np.zeros(num_rays, dtype=float)
    failures = 0
    start_time = time.perf_counter()

    for index in range(num_rays):
        initial_coord = np.array([start_row, rng.choice(start_cols)])
        try:
            ray_stats = []
            _, _, coord, _, _, power = single_ray_opaque_simulate(
                med,
                initial_direction,
                initial_coord=initial_coord,
                max_itr=max_irt,
                beta=beta,
                grad_field=grad_field,
                power_tolerance=power_tolerance,
                ray_stats=ray_stats,
            )
            if ray_stats and ray_stats[-1]["exit_col"] == core.EXIT_REFLECT:
                values[index] = power
        except AssertionError:
            failures += 1

    return values, failures, time.perf_counter() - start_time

# 95% confidence interval size
def probability_ci95(probability: float, sample_size: int) -> float:
    return 1.96 * math.sqrt(probability * (1.0 - probability) / sample_size)


def mean_ci95(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    return 1.96 * np.std(values, ddof=1) / math.sqrt(len(values))

def run_one_case(medium_kind: str, density: float, medium_seed: int) -> dict[str, object]:
    smp_seed = 10000 + medium_seed
    mcrt_probability_seed = 20000 + medium_seed
    mcrt_energy_seed = 30000 + medium_seed

    print(
        f"\n{medium_kind} | density={density:.2f} | medium_seed={medium_seed}",
        flush=True,
    )

    med = make_medium(medium_kind, density, medium_seed)
    actual_density = float(np.mean(med))

    grad_field = reliable_gradient_field(
        med,
        smooth_sigma=smooth_sigma,
        normalize=True,
    )

    ray_stats, smp_failures, smp_seconds = build_smp_path_database(
        med,
        grad_field,
        smp_rays,
        smp_seed,
    )
    smp_probability, smp_energy = smp_outputs(ray_stats, beta)

    mcrt_probability_values, mcrt_probability_failures, mcrt_probability_seconds = run_mcrt(
        med,
        grad_field,
        beta=0.0,
        num_rays=mcrt_rays,
        seed=mcrt_probability_seed,
    )
    mcrt_energy_values, mcrt_energy_failures, mcrt_energy_seconds = run_mcrt(
        med,
        grad_field,
        beta=beta,
        num_rays=mcrt_rays,
        seed=mcrt_energy_seed,
    )

    mcrt_probability = float(np.count_nonzero(mcrt_probability_values) / mcrt_rays)
    mcrt_energy = float(np.mean(mcrt_energy_values))

    probability_error = smp_probability - mcrt_probability
    energy_error = smp_energy - mcrt_energy

    print(
        f"  probability error = {100.0 * probability_error:.3f} pp | "
        f"energy error = {100.0 * energy_error:.5f} pp",
        flush=True,
    )

    return {
        "medium_kind": medium_kind,
        "target_density": density,
        "actual_density": actual_density,
        "medium_seed": medium_seed,
        "grid_size": grid_size,
        "beta": beta,
        "num_smp_rays": smp_rays,
        "num_mcrt_rays": mcrt_rays,
        "smooth_sigma": smooth_sigma,
        "smp_probability": smp_probability,
        "mcrt_probability": mcrt_probability,
        "smp_probability_percent": 100.0 * smp_probability,
        "mcrt_probability_percent": 100.0 * mcrt_probability,
        "smp_probability_ci95_pp": 100.0 * probability_ci95(smp_probability, smp_rays),
        "mcrt_probability_ci95_pp": 100.0 * probability_ci95(mcrt_probability, mcrt_rays),
        "probability_error_pp": 100.0 * probability_error,
        "probability_abs_error_pp": 100.0 * abs(probability_error),
        "smp_energy": smp_energy,
        "mcrt_energy": mcrt_energy,
        "smp_energy_percent": 100.0 * smp_energy,
        "mcrt_energy_percent": 100.0 * mcrt_energy,
        "energy_error_pp": 100.0 * energy_error,
        "energy_abs_error_pp": 100.0 * abs(energy_error),
        "smp_failures": smp_failures,
        "mcrt_probability_failures": mcrt_probability_failures,
        "mcrt_energy_failures": mcrt_energy_failures,
        "smp_seconds": smp_seconds,
        "mcrt_probability_seconds": mcrt_probability_seconds,
        "mcrt_energy_seconds": mcrt_energy_seconds,
    }


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else math.nan


def rmse(values: list[float]) -> float:
    return math.sqrt(mean([value * value for value in values]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
#     summary_rows = []
#
#     for medium_kind in media:
#         for density in densities:
#             group = [
#                 row for row in rows
#                 if row["medium_kind"] == medium_kind
#                 and row["target_density"] == density
#             ]
#
#             probability_errors = [float(row["probability_error_pp"]) for row in group]
#             energy_errors = [float(row["energy_error_pp"]) for row in group]
#             smp_probabilities = [float(row["smp_probability_percent"]) for row in group]
#             mcrt_probabilities = [float(row["mcrt_probability_percent"]) for row in group]
#             smp_energies = [float(row["smp_energy_percent"]) for row in group]
#             mcrt_energies = [float(row["mcrt_energy_percent"]) for row in group]
#
#             summary_rows.append({
#                 "medium_kind": medium_kind,
#                 "target_density": density,
#                 "n_realizations": len(group),
#                 "mean_actual_density": mean([float(row["actual_density"]) for row in group]),
#                 "mean_smp_probability_percent": mean(smp_probabilities),
#                 "mean_mcrt_probability_percent": mean(mcrt_probabilities),
#                 "std_smp_probability_percent": std(smp_probabilities),
#                 "std_mcrt_probability_percent": std(mcrt_probabilities),
#                 "ci95_smp_probability_percent": mean_ci95(smp_probabilities),
#                 "ci95_mcrt_probability_percent": mean_ci95(mcrt_probabilities),
#                 "bias_probability_error_pp": mean(probability_errors),
#                 "mean_abs_probability_error_pp": mean([abs(value) for value in probability_errors]),
#                 "rmse_probability_error_pp": rmse(probability_errors),
#                 "std_probability_error_pp": std(probability_errors),
#                 "mean_smp_energy_percent": mean(smp_energies),
#                 "mean_mcrt_energy_percent": mean(mcrt_energies),
#                 "std_smp_energy_percent": std(smp_energies),
#                 "std_mcrt_energy_percent": std(mcrt_energies),
#                 "ci95_smp_energy_percent": mean_ci95(smp_energies),
#                 "ci95_mcrt_energy_percent": mean_ci95(mcrt_energies),
#                 "bias_energy_error_pp": mean(energy_errors),
#                 "mean_abs_energy_error_pp": mean([abs(value) for value in energy_errors]),
#                 "rmse_energy_error_pp": rmse(energy_errors),
#                 "std_energy_error_pp": std(energy_errors),
#                 "mean_smp_seconds": mean([float(row["smp_seconds"]) for row in group]),
#                 "mean_mcrt_probability_seconds": mean([
#                     float(row["mcrt_probability_seconds"]) for row in group
#                 ]),
#                 "mean_mcrt_energy_seconds": mean([
#                     float(row["mcrt_energy_seconds"]) for row in group
#                 ]),
#             })
#
#     write_csv(path, summary_rows)


def plot_realization_summary(summary_path: Path) -> None:
    plot_dir = OUTDIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    with summary_path.open("r", newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))

    for medium_kind in media:
        rows = [
            row for row in summary_rows
            if row["medium_kind"] == medium_kind
        ]
        rows.sort(key=lambda row: float(row["target_density"]))

        x = np.array([float(row["target_density"]) for row in rows])

        smp_probability = np.array([
            float(row["mean_smp_probability_percent"]) for row in rows
        ])
        mcrt_probability = np.array([
            float(row["mean_mcrt_probability_percent"]) for row in rows
        ])
        smp_probability_ci = np.array([
            float(row["ci95_smp_probability_percent"]) for row in rows
        ])
        mcrt_probability_ci = np.array([
            float(row["ci95_mcrt_probability_percent"]) for row in rows
        ])

        smp_energy = np.array([
            float(row["mean_smp_energy_percent"]) for row in rows
        ])
        mcrt_energy = np.array([
            float(row["mean_mcrt_energy_percent"]) for row in rows
        ])
        smp_energy_ci = np.array([
            float(row["ci95_smp_energy_percent"]) for row in rows
        ])
        mcrt_energy_ci = np.array([
            float(row["ci95_mcrt_energy_percent"]) for row in rows
        ])

        probability_error = np.array([
            float(row["bias_probability_error_pp"]) for row in rows
        ])
        energy_error = np.array([
            float(row["bias_energy_error_pp"]) for row in rows
        ])
        probability_error_std = np.array([
            float(row["std_probability_error_pp"]) for row in rows
        ])
        energy_error_std = np.array([
            float(row["std_energy_error_pp"]) for row in rows
        ])

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

        axes[0, 0].errorbar(
            x, smp_probability, yerr=smp_probability_ci,
            marker="o", capsize=4, label="SMP"
        )
        axes[0, 0].errorbar(
            x, mcrt_probability, yerr=mcrt_probability_ci,
            marker="s", capsize=4, label="MCRT"
        )
        axes[0, 0].set_title("Reflection probability")
        axes[0, 0].set_ylabel("Probability (%)")
        axes[0, 0].legend()

        axes[0, 1].errorbar(
            x, smp_energy, yerr=smp_energy_ci,
            marker="o", capsize=4, label="SMP"
        )
        axes[0, 1].errorbar(
            x, mcrt_energy, yerr=mcrt_energy_ci,
            marker="s", capsize=4, label="MCRT"
        )
        axes[0, 1].set_title("Energy reflectivity")
        axes[0, 1].set_ylabel("Energy reflectivity (%)")
        axes[0, 1].legend()

        axes[1, 0].axhline(0.0, color="black", linewidth=1)
        axes[1, 0].errorbar(
            x, probability_error, yerr=probability_error_std,
            marker="o", capsize=4, color="tab:red"
        )
        axes[1, 0].set_title("Probability error")
        axes[1, 0].set_ylabel("SMP - MCRT (percentage points)")

        axes[1, 1].axhline(0.0, color="black", linewidth=1)
        axes[1, 1].errorbar(
            x, energy_error, yerr=energy_error_std,
            marker="o", capsize=4, color="tab:red"
        )
        axes[1, 1].set_title("Energy error")
        axes[1, 1].set_ylabel("SMP - MCRT (percentage points)")

        for ax in axes.flat:
            ax.set_xlabel("Target solid area fraction")
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"{medium_kind.capitalize()} media: multiple realizations per area fraction",
            fontsize=14,
        )

        out_path = plot_dir / f"{medium_kind}_realizations_summary.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {out_path}")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    rows = []
    total = len(media) * len(densities) * len(medium_seeds)
    case_count = 0

    print(f"Running {total} medium-realization cases.")

    for medium_kind in media:
        for density in densities:
            for medium_seed in medium_seeds:
                case_count += 1
                print(f"\nCase {case_count}/{total}", flush=True)
                rows.append(run_one_case(medium_kind, density, medium_seed))

    runs_path = OUTDIR / "realization_runs.csv"
    summary_path = OUTDIR / "realization_summary.csv"

    write_csv(runs_path, rows)
    # write_summary(summary_path, rows)
    plot_realization_summary(summary_path)


if __name__ == "__main__":
    main()
