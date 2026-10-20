import argparse
import numpy as np
from core import get_bin_index
from media import circle_medium_nonoverlap, square_medium_nonoverlap
from simulate import simulate_onesided_reflection


def make_medium(medium_kind, density, seed):
    if medium_kind == "square":
        return square_medium_nonoverlap(500, 20, density, seed=seed)

    if medium_kind == "circle":
        return circle_medium_nonoverlap(500, 500, radius=10, density=density, seed=seed)


def medium_title(medium_kind):
    return "Square-particle medium" if medium_kind == "square" else "Circular-particle medium"


def calculate_theoretical_reflection_probability( # geometric ray count
    P_matrix,
    initial_direction=np.array([-1.0, 0.0])
):
    num_bins = 8
    reflect_col = 8

    Q = P_matrix[:, :num_bins]
    r = P_matrix[:, reflect_col]

    initial_bin = get_bin_index(initial_direction, num_bins)

    rho = max(abs(np.linalg.eigvals(Q)))
    if rho >= 1:
        raise ValueError("Geometric renewal matrix is non-absorbing.")

    reflect_prob_by_bin = np.linalg.solve(np.eye(num_bins) - Q, r)
    return reflect_prob_by_bin[initial_bin]


def calculate_theoretical_energy_reflectivity(
    P_matrix,
    erlang_data,
    beta,
    initial_direction=np.array([-1.0, 0.0]),
):
    num_bins = 8
    reflect_col = 8

    Q = P_matrix[:, :num_bins]
    r = P_matrix[:, reflect_col]
    initial_bin = get_bin_index(initial_direction, num_bins)

    # each transition receives its own chord-length attenuation factor.
    phi = np.ones_like(P_matrix, dtype=float)

    for incoming_bin in range(num_bins):
        for outgoing_col in range(P_matrix.shape[1]):
            params = erlang_data[(incoming_bin, outgoing_col)]
            k = params["k"]
            theta = params["theta"]

            # Laplace transform of Gamma/Erlang(k, theta) at s = 2*beta.
            phi[incoming_bin, outgoing_col] = (
                1.0 + 2.0 * beta * theta
            ) ** (-k)

    # continuation transitions: i -> j, j in direction bins.
    A = Q * phi[:, :num_bins]

    # reflected exit transitions: i -> EXIT_REFLECT.
    b = r * phi[:, reflect_col]

    rho = max(abs(np.linalg.eigvals(A)))
    if rho >= 1:
        raise ValueError("Energy renewal matrix is non-absorbing.")

    reflect_energy_by_bin = np.linalg.solve(np.eye(num_bins) - A, b)
    return reflect_energy_by_bin[initial_bin]


def run_validation(medium_kind):
    import time
    from core import RefractiveIndexConst
    from gradient import reliable_gradient_field
    from simulate import SimConfig
    from smp_simulator import simulator_opaque

    num_rays = 20000
    beta = 0.1

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

    print(f"\n {medium_title(medium_kind)} ")

    # trace beta=0 geometry once and store ray path stats.
    smp_build_t0 = time.time()
    sim = simulator_opaque(cfg, num_direction_bins=8)
    sim.sim_opaque(
        num_sim=num_rays,
        collect_chords=False,
        collect_ray_stats=True,
    )
    smp_build_time = time.time() - smp_build_t0

    # smp probability at beta=0.0.
    smp_probability_t0 = time.time()
    smp_probability = sim.get_pathwise_reflection_probability()
    smp_probability_time = time.time() - smp_probability_t0

    # smp energy reflectivity at beta=0.1.
    smp_energy_t0 = time.time()
    smp_energy = sim.get_pathwise_energy_reflectivity(beta)
    smp_energy_time = time.time() - smp_energy_t0

    # MCRT probability at beta=0.0.
    mcrt_probability_t0 = time.time()
    vals_prob, _ = simulate_onesided_reflection(med, beta=0.0, num_sim=num_rays)
    mcrt_probability_time = time.time() - mcrt_probability_t0

    mcrt_probability = np.count_nonzero(vals_prob) / len(vals_prob)

    # MCRT energy reflectivity at beta=0.1.
    mcrt_energy_t0 = time.time()
    vals_energy, _ = simulate_onesided_reflection(med, beta=beta, num_sim=num_rays)
    mcrt_energy_time = time.time() - mcrt_energy_t0

    mcrt_energy = np.mean(vals_energy)

    print("\n--- reflection probability ---")
    print(f"smp reflection probability: {smp_probability * 100:.2f}%")
    print(f"mcrt reflection probability: {mcrt_probability * 100:.2f}%")
    print_accuracy_metrics(
        "Reflection probability",
        np.array([smp_probability]),
        np.array([mcrt_probability]),
    )

    print("\n--- energy reflectivity ---")
    print(f"smp energy reflectivity: {smp_energy * 100:.6f}%")
    print(f"mcrt energy reflectivity: {mcrt_energy * 100:.6f}%")
    print_accuracy_metrics(
        "Energy reflectivity",
        np.array([smp_energy]),
        np.array([mcrt_energy]),
    )

    print("\n--- timing ---")
    print(f"smp build time: {smp_build_time:.4f} seconds")
    print(f"smp probability evaluation time: {smp_probability_time:.8f} seconds")
    print(f"smp probability full time: {smp_build_time + smp_probability_time:.4f} seconds")
    print(f"mcrt probability run time: {mcrt_probability_time:.4f} seconds")

    print(f"smp energy evaluation time: {smp_energy_time:.8f} seconds")
    print(f"smp energy full time: {smp_build_time + smp_energy_time:.4f} seconds")
    print(f"mcrt energy run time: {mcrt_energy_time:.4f} seconds")

def print_accuracy_metrics(label, model_values, mcrt_values):
    residual = model_values - mcrt_values
    mae = np.mean(np.abs(residual))
    mse = np.mean(residual**2)
    rmse = np.sqrt(mse)
    max_abs = np.max(np.abs(residual))
    bias = np.mean(residual)

    print(f"\n{label} accuracy metrics ")
    print(f"MAE: {mae:.8e}")
    print(f"MSE:{mse:.8e}")
    print(f"RMSE: {rmse:.8e}")
    print(f"Max absolute residual: {max_abs:.8e}")
    print(f"Bias/model - MCRT:{bias:.8e}")


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
        run_validation(medium_kind)
