import argparse
import os
import numpy as np
from pettingzoo.classic import rps_v2, leduc_holdem_v4
from pettingzoo.atari import boxing_v2
import random

import ray
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import (
    PPO,
    PPOConfig,
    PPOTF1Policy,
    PPOTF2Policy,
    PPOTorchPolicy,
)
from ray.rllib.env import PettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env, register_trainable
from ray.tune.registry import get_trainable_cls

from mmd import MMDAPPO, OldMMDAPPO, MMDPPO

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=150, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward",
    type=float,
    default=1000.0,
    help="Reward at which we stop training.",
)

def env_creator(args):
    env = rps_v2.env(num_actions=5, max_cycles=15)
    return env

register_trainable("MMDAPPO", MMDAPPO)
register_trainable("OldMMDAPPO", OldMMDAPPO)
register_trainable("MMDPPO", MMDPPO)
register_env("NashEnv", lambda config: PettingZooEnv(env_creator(config)))

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from collections import defaultdict

class ActionDistributionCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()

    def on_postprocess_trajectory(self, *, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
        # Get the actions taken in the current batch
        actions = postprocessed_batch['actions']

        # Track the count of each action within the episode
        action_counts = defaultdict(int)
        for action in actions:
            action_counts[action] += 1

        # Log the action counts as custom metrics
        for action, count in action_counts.items():
            # Record each action's count as a custom metric for the episode
            episode.custom_metrics[f"action_{action}_count"] = count

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # Optionally, you could further process or log the data at the end of the episode
        pass


def run_same_policy(args, stop):
    """Use the same policy for both agents (trivial case)."""
    config = (
        get_trainable_cls("PPO")
        .get_default_config()
        .environment("NashEnv")
        .framework(args.framework)
        .callbacks(ActionDistributionCallback)
        .env_runners(
            num_env_runners=8
        )
        .learners(
            num_gpus_per_learner=1
        )
        .training(
            use_kl_loss=True,
        )
        .reporting(
            min_time_s_per_iteration=30,
        )
    )
    wandb_logger = WandbLoggerCallback(
        project="rock-paper-scissors",
        entity="tmaidment",
        group=None,
        job_type="training",
        config=config,
    )

    results = tune.Tuner(
        "MMDPPO",
        param_space=config,
        run_config=air.RunConfig(stop=stop, callbacks=[wandb_logger], verbose=1)
    ).fit()

    if args.as_test:
        # Check vs 0.0 as we are playing a zero-sum game.
        check_learning_achieved(results, 0.0)


if __name__ == "__main__":
    args = parser.parse_args()

    ray.init()

    # stop = {
    #     "training_iteration": args.stop_iters,
    #     "timesteps_total": args.stop_timesteps,
    #     "episode_reward_mean": args.stop_reward,
    # }

    run_same_policy(args, stop=None)
