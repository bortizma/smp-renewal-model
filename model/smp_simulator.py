import numpy as np
from simulate import SimConfig
from core import (
    single_ray_opaque_simulate,
    EXIT_REFLECT,
    NUM_DIRECTION_BINS,
    NUM_TRANSITION_COLS,
)
import logging
logging.basicConfig(level=logging.WARNING)


class simulator_opaque:
    def __init__(self, sim_cfg: SimConfig, num_direction_bins=NUM_DIRECTION_BINS):
        self.sim_cfg = sim_cfg
        self.num_direction_bins = num_direction_bins
        self.transition_matrix_opaque = np.zeros((num_direction_bins, NUM_TRANSITION_COLS))
        self.chord_data_opaque = {
            (i, j): []
            for i in range(num_direction_bins)
            for j in range(NUM_TRANSITION_COLS)
        }
        self.ray_stats_opaque = []

    def sim_opaque(self, num_sim, collect_chords=True, collect_ray_stats=True):  # probability runs skip path-length storage.
        for _ in range(num_sim):
            while True:
                start_row = self.sim_cfg.medium.shape[0] -2
                start_col = np.random.choice(np.arange(200, self.sim_cfg.medium.shape[1] - 200))

                if self.sim_cfg.medium[start_row, start_col] == 0:
                    initial_coord = np.array([start_row, start_col])
                    break
            # og simulation
            path, itr, coord, direction, strike_points, power = single_ray_opaque_simulate(
                self.sim_cfg.medium,
                initial_direction = self.sim_cfg.initial_direction,
                initial_coord = initial_coord,
                max_itr = self.sim_cfg.max_itr,
                beta = self.sim_cfg.beta,
                grad_field = self.sim_cfg.grad_field,
                transition_matrix = self.transition_matrix_opaque,
                chord_data=self.chord_data_opaque if collect_chords else None,
                ray_stats=self.ray_stats_opaque if collect_ray_stats else None
            )

    def get_transition_probabilities(self):
        # normalize transition matrix
        row_sums = self.transition_matrix_opaque.sum(axis=1, keepdims=True)
        return np.divide(self.transition_matrix_opaque, row_sums,
                         out=np.zeros_like(self.transition_matrix_opaque),
                         where=row_sums != 0)

    def fit_erlang_distributions(self):
        #not used, past work
        """
        fits one Erlang/Gamma distribution per transition (i, j), where
        i is the incoming direction bin and j is either an outgoing direction
        bin or an absorbing exit column.
        """
        erlang_params = {}

        for transition, lengths in self.chord_data_opaque.items():
            valid_lengths = np.asarray(lengths, dtype=float)
            valid_lengths = valid_lengths[valid_lengths > 0]

            if len(valid_lengths) >= 2:
                mean_val = float(np.mean(valid_lengths))
                var_val = float(np.var(valid_lengths, ddof=1))

                if var_val > 0:
                    k = max(1, int(round(mean_val ** 2 / var_val)))
                    theta = mean_val / k
                else:
                    k = 1
                    theta = mean_val

                erlang_params[transition] = {
                    "k": k,
                    "theta": theta,
                    "mean": mean_val,
                    "n": len(valid_lengths),
                }

            elif len(valid_lengths) == 1:
                erlang_params[transition] = {
                    "k": 1,
                    "theta": float(valid_lengths[0]),
                    "mean": float(valid_lengths[0]),
                    "n": 1,
                }

            else:
                erlang_params[transition] = {
                    "k": 1,
                    "theta": 0.0,
                    "mean": 0.0,
                    "n": 0,
                }

        return erlang_params

    def get_pathwise_reflection_probability(self):
        if not self.ray_stats_opaque:
            return 0.0

        reflected = sum(
            stat["exit_col"] == EXIT_REFLECT
            for stat in self.ray_stats_opaque
        )
        return reflected / len(self.ray_stats_opaque)

    def get_pathwise_energy_reflectivity(self, beta):
        if not self.ray_stats_opaque:
            return 0.0

        total = 0.0
        for stat in self.ray_stats_opaque:
            if stat["exit_col"] == EXIT_REFLECT:
                total += np.exp(-2.0 * beta * stat["path_length"])

        return total / len(self.ray_stats_opaque)
