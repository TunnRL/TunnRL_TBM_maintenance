"""Defining callbacks to use in model training.
"""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class StopTrainingOnNoModelImprovement(BaseCallback):
    """
    Stop the training early if there is no new best model (new best mean reward) after more than N consecutive evaluations.

    It is possible to define a minimum number of evaluations before start to count evaluations without improvement.

    It must be used with the ``EvalCallback``.

    :param max_no_improvement_evals: Maximum number of consecutive evaluations without a new best model.
    :param min_evals: Number of evaluations before start to count evaluations without improvements.
    :param verbose: Verbosity of the output (set to 1 for info messages)
    """

    def __init__(self, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.last_best_mean_reward = -np.inf
        self.no_improvement_evals = 0

    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnNoModelImprovement`` callback must be used with an ``EvalCallback``"

        continue_training = True

        if self.n_calls > self.min_evals:
            if self.parent.best_mean_reward > self.last_best_mean_reward:
                self.no_improvement_evals = 0
            else:
                self.no_improvement_evals += 1
                if self.no_improvement_evals > self.max_no_improvement_evals:
                    continue_training = False

        self.last_best_mean_reward = self.parent.best_mean_reward

        if self.verbose > 0 and not continue_training:
            print(
                f"Stopping training because there was no new best model in the last {self.no_improvement_evals:d} evaluations"
            )

        return continue_training
