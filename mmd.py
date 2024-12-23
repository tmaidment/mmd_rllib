import gymnasium as gym
import torch
import numpy as np
import logging
from typing import Any, Dict, List, Optional, Type, Union
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.appo.appo_torch_policy import APPOTorchPolicy
from ray.rllib.utils.torch_utils import explained_variance, sequence_mask
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.numpy import convert_to_numpy
from typing import Dict, Union, Type
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.appo import APPO
from ray.rllib.utils.torch_utils import apply_grad_clipping, explained_variance
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
import ray.rllib.algorithms.impala.vtrace_torch as vtrace
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.annotations import override
import math
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    global_norm,
    sequence_mask,
)
from ray.rllib.models.torch.torch_action_dist import (
    TorchDistributionWrapper,
    TorchCategorical,
)
from ray.rllib.algorithms.impala.impala_torch_policy import (
    make_time_major,
    VTraceOptimizer,
)
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

SMALL_POSITIVE = 1e-10
DEFAULT_VALUE = 1
WARMUP_ITERATIONS = 10


class EMAgnetTorchPolicy(APPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        # MMD-specific attributes
        self.temp = config.get("temperature_schedule", lambda t: 0.01)
        self.mag_lr = config.get("magnet_learning_rate_schedule", lambda t: 0.005)
        self.magnet_policy = None
        self.iteration = 0

        super().__init__(observation_space, action_space, config)

    @override(APPOTorchPolicy)
    def make_model(self) -> ModelV2:
        self.model = super().make_model()
        self.magnet_policy = super().make_model()
        return self.model

    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Constructs the loss for APPO.

        With IS modifications and V-trace for Advantage Estimation.

        Args:
            model (ModelV2): The Model to calculate the loss for.
            dist_class (Type[ActionDistribution]): The action distr. class.
            train_batch: The training data.

        Returns:
            Union[TensorType, List[TensorType]]: A single loss tensor or a list
                of loss tensors.
        """
        device = next(self.model.parameters()).device
        self.magnet_policy = self.magnet_policy.to(device)

        target_model = self.target_models[model]

        model_out, _ = model(train_batch)
        action_dist = dist_class(model_out, model)

        if isinstance(self.action_space, gym.spaces.Discrete):
            is_multidiscrete = False
            output_hidden_shape = [self.action_space.n]
        elif isinstance(self.action_space, gym.spaces.multi_discrete.MultiDiscrete):
            is_multidiscrete = True
            output_hidden_shape = self.action_space.nvec.astype(np.int32)
        else:
            is_multidiscrete = False
            output_hidden_shape = 1

        def _make_time_major(*args, **kwargs):
            return make_time_major(
                self, train_batch.get(SampleBatch.SEQ_LENS), *args, **kwargs
            )

        actions = train_batch[SampleBatch.ACTIONS]
        dones = train_batch[SampleBatch.TERMINATEDS]
        rewards = train_batch[SampleBatch.REWARDS]
        behaviour_logits = train_batch[SampleBatch.ACTION_DIST_INPUTS]

        target_model_out, _ = target_model(train_batch)

        prev_action_dist = dist_class(behaviour_logits, model)
        values = model.value_function()
        values_time_major = _make_time_major(values)
        bootstrap_values_time_major = _make_time_major(
            train_batch[SampleBatch.VALUES_BOOTSTRAPPED]
        )
        bootstrap_value = bootstrap_values_time_major[-1]

        if self.is_recurrent():
            max_seq_len = torch.max(train_batch[SampleBatch.SEQ_LENS])
            mask = sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
            mask = torch.reshape(mask, [-1])
            mask = _make_time_major(mask)
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        else:
            reduce_mean_valid = torch.mean

        if self.config["vtrace"]:
            logger.debug("Using V-Trace surrogate loss (vtrace=True)")

            old_policy_behaviour_logits = target_model_out.detach()
            old_policy_action_dist = dist_class(old_policy_behaviour_logits, model)

            if isinstance(output_hidden_shape, (list, tuple, np.ndarray)):
                unpacked_behaviour_logits = torch.split(
                    behaviour_logits, list(output_hidden_shape), dim=1
                )
                unpacked_old_policy_behaviour_logits = torch.split(
                    old_policy_behaviour_logits, list(output_hidden_shape), dim=1
                )
            else:
                unpacked_behaviour_logits = torch.chunk(
                    behaviour_logits, output_hidden_shape, dim=1
                )
                unpacked_old_policy_behaviour_logits = torch.chunk(
                    old_policy_behaviour_logits, output_hidden_shape, dim=1
                )

            # Prepare actions for loss.
            loss_actions = (
                actions if is_multidiscrete else torch.unsqueeze(actions, dim=1)
            )

            # Prepare KL for loss.
            action_kl = _make_time_major(old_policy_action_dist.kl(action_dist))
            # reverse_action_kl = _make_time_major(action_dist.kl(old_policy_action_dist))

            # Compute vtrace on the CPU for better perf.
            vtrace_returns = vtrace.multi_from_logits(
                behaviour_policy_logits=_make_time_major(unpacked_behaviour_logits),
                target_policy_logits=_make_time_major(
                    unpacked_old_policy_behaviour_logits
                ),
                actions=torch.unbind(_make_time_major(loss_actions), dim=2),
                discounts=(1.0 - _make_time_major(dones).float())
                * self.config["gamma"],
                rewards=_make_time_major(rewards),
                values=values_time_major,
                bootstrap_value=bootstrap_value,
                dist_class=TorchCategorical if is_multidiscrete else dist_class,
                model=model,
                clip_rho_threshold=self.config["vtrace_clip_rho_threshold"],
                clip_pg_rho_threshold=self.config["vtrace_clip_pg_rho_threshold"],
            )

            actions_logp = _make_time_major(action_dist.logp(actions))
            prev_actions_logp = _make_time_major(prev_action_dist.logp(actions))
            old_policy_actions_logp = _make_time_major(
                old_policy_action_dist.logp(actions)
            )
            is_ratio = torch.clamp(
                torch.exp(prev_actions_logp - old_policy_actions_logp), 0.0, 2.0
            )
            logp_ratio = is_ratio * torch.exp(actions_logp - prev_actions_logp)
            self._is_ratio = is_ratio

            advantages = vtrace_returns.pg_advantages.to(logp_ratio.device)
            surrogate_loss = torch.min(
                advantages * logp_ratio,
                advantages
                * torch.clamp(
                    logp_ratio,
                    1 - self.config["clip_param"],
                    1 + self.config["clip_param"],
                ),
            )

            mean_kl_loss = reduce_mean_valid(action_kl)
            # mean_kl_loss = mean_reverse_kl_loss = reduce_mean_valid(reverse_action_kl)
            mean_policy_loss = -reduce_mean_valid(surrogate_loss)

            # The value function loss.
            value_targets = vtrace_returns.vs.to(values_time_major.device)
            delta = values_time_major - value_targets
            mean_vf_loss = 0.5 * reduce_mean_valid(torch.pow(delta, 2.0))

            # The entropy loss.
            mean_entropy = reduce_mean_valid(_make_time_major(action_dist.entropy()))

        else:
            logger.debug("Using PPO surrogate loss (vtrace=False)")

            # Prepare KL for Loss
            action_kl = _make_time_major(prev_action_dist.kl(action_dist))
            actions_logp = _make_time_major(action_dist.logp(actions))
            prev_actions_logp = _make_time_major(prev_action_dist.logp(actions))
            logp_ratio = torch.exp(actions_logp - prev_actions_logp)

            advantages = _make_time_major(train_batch[Postprocessing.ADVANTAGES])
            surrogate_loss = torch.min(
                advantages * logp_ratio,
                advantages
                * torch.clamp(
                    logp_ratio,
                    1 - self.config["clip_param"],
                    1 + self.config["clip_param"],
                ),
            )

            mean_kl_loss = reduce_mean_valid(action_kl)
            # mean_kl_loss = mean_reverse_kl_loss = reduce_mean_valid(reverse_action_kl)
            mean_policy_loss = -reduce_mean_valid(surrogate_loss)

            # The value function loss.
            value_targets = _make_time_major(train_batch[Postprocessing.VALUE_TARGETS])
            delta = values_time_major - value_targets
            mean_vf_loss = 0.5 * reduce_mean_valid(torch.pow(delta, 2.0))

            # The entropy loss.
            mean_entropy = reduce_mean_valid(_make_time_major(action_dist.entropy()))

        # The summed weighted loss.
        total_loss = mean_policy_loss - mean_entropy * self.entropy_coeff
        # Optional additional KL Loss
        if self.config["use_kl_loss"]:
            total_loss += self.kl_coeff * mean_kl_loss

        # Compute outputs from magnet policy
        magnet_out, _ = self.magnet_policy(train_batch)
        magnet_dist = dist_class(magnet_out, self.magnet_policy)

        # KL divergence between current policy and magnet policy
        kl_div = action_dist.kl(magnet_dist)

        # MMD loss
        temp = self.temp(self.iteration)
        mmd_loss = temp * torch.mean(kl_div)

        # Add MMD loss to the total loss
        total_loss += mmd_loss

        # Optional vf loss (or in a separate term due to separate
        # optimizers/networks).
        loss_wo_vf = total_loss
        if not self.config["_separate_vf_optimizer"]:
            total_loss += mean_vf_loss * self.config["vf_loss_coeff"]

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = mean_policy_loss
        model.tower_stats["mean_kl_loss"] = mean_kl_loss
        model.tower_stats["mmd_loss"] = mmd_loss
        model.tower_stats["kl_div"] = torch.mean(kl_div)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["value_targets"] = value_targets
        model.tower_stats["vf_explained_var"] = explained_variance(
            torch.reshape(value_targets, [-1]),
            torch.reshape(values_time_major, [-1]),
        )

        # Perform the magnet update
        with torch.no_grad():
            for param, magnet_param in zip(
                self.model.parameters(), self.magnet_policy.parameters()
            ):
                magnet_param.data = (
                    self.mag_lr(self.iteration) * param.data
                    + (1 - self.mag_lr(self.iteration)) * magnet_param.data
                )

        # Return one total loss or two losses: vf vs rest (policy + kl).
        if self.config["_separate_vf_optimizer"]:
            return loss_wo_vf, mean_vf_loss
        else:
            return total_loss


class EMAgnetAPPO(APPO):
    def get_default_policy_class(cls, config):
        return EMAgnetTorchPolicy


class MMDTorchPolicy(APPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        assert config["use_kl_loss"] == True, "use_kl_loss must be True for MMD"

        super().__init__(observation_space, action_space, config)

    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Constructs the loss for APPO.

        With IS modifications and V-trace for Advantage Estimation.

        Args:
            model (ModelV2): The Model to calculate the loss for.
            dist_class (Type[ActionDistribution]): The action distr. class.
            train_batch: The training data.

        Returns:
            Union[TensorType, List[TensorType]]: A single loss tensor or a list
                of loss tensors.
        """
        target_model = self.target_models[model]

        model_out, _ = model(train_batch)
        action_dist = dist_class(model_out, model)

        if isinstance(self.action_space, gym.spaces.Discrete):
            is_multidiscrete = False
            output_hidden_shape = [self.action_space.n]
        elif isinstance(self.action_space, gym.spaces.multi_discrete.MultiDiscrete):
            is_multidiscrete = True
            output_hidden_shape = self.action_space.nvec.astype(np.int32)
        else:
            is_multidiscrete = False
            output_hidden_shape = 1

        def _make_time_major(*args, **kwargs):
            return make_time_major(
                self, train_batch.get(SampleBatch.SEQ_LENS), *args, **kwargs
            )

        actions = train_batch[SampleBatch.ACTIONS]
        dones = train_batch[SampleBatch.TERMINATEDS]
        rewards = train_batch[SampleBatch.REWARDS]
        behaviour_logits = train_batch[SampleBatch.ACTION_DIST_INPUTS]

        target_model_out, _ = target_model(train_batch)

        prev_action_dist = dist_class(behaviour_logits, model)
        values = model.value_function()
        values_time_major = _make_time_major(values)
        bootstrap_values_time_major = _make_time_major(
            train_batch[SampleBatch.VALUES_BOOTSTRAPPED]
        )
        bootstrap_value = bootstrap_values_time_major[-1]

        if self.is_recurrent():
            max_seq_len = torch.max(train_batch[SampleBatch.SEQ_LENS])
            mask = sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
            mask = torch.reshape(mask, [-1])
            mask = _make_time_major(mask)
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        else:
            reduce_mean_valid = torch.mean

        if self.config["vtrace"]:
            logger.debug("Using V-Trace surrogate loss (vtrace=True)")

            old_policy_behaviour_logits = target_model_out.detach()
            old_policy_action_dist = dist_class(old_policy_behaviour_logits, model)

            if isinstance(output_hidden_shape, (list, tuple, np.ndarray)):
                unpacked_behaviour_logits = torch.split(
                    behaviour_logits, list(output_hidden_shape), dim=1
                )
                unpacked_old_policy_behaviour_logits = torch.split(
                    old_policy_behaviour_logits, list(output_hidden_shape), dim=1
                )
            else:
                unpacked_behaviour_logits = torch.chunk(
                    behaviour_logits, output_hidden_shape, dim=1
                )
                unpacked_old_policy_behaviour_logits = torch.chunk(
                    old_policy_behaviour_logits, output_hidden_shape, dim=1
                )

            # Prepare actions for loss.
            loss_actions = (
                actions if is_multidiscrete else torch.unsqueeze(actions, dim=1)
            )

            # Prepare KL for loss.
            action_kl = _make_time_major(old_policy_action_dist.kl(action_dist))
            reverse_action_kl = _make_time_major(action_dist.kl(old_policy_action_dist))

            # Compute vtrace on the CPU for better perf.
            vtrace_returns = vtrace.multi_from_logits(
                behaviour_policy_logits=_make_time_major(unpacked_behaviour_logits),
                target_policy_logits=_make_time_major(
                    unpacked_old_policy_behaviour_logits
                ),
                actions=torch.unbind(_make_time_major(loss_actions), dim=2),
                discounts=(1.0 - _make_time_major(dones).float())
                * self.config["gamma"],
                rewards=_make_time_major(rewards),
                values=values_time_major,
                bootstrap_value=bootstrap_value,
                dist_class=TorchCategorical if is_multidiscrete else dist_class,
                model=model,
                clip_rho_threshold=self.config["vtrace_clip_rho_threshold"],
                clip_pg_rho_threshold=self.config["vtrace_clip_pg_rho_threshold"],
            )

            actions_logp = _make_time_major(action_dist.logp(actions))
            prev_actions_logp = _make_time_major(prev_action_dist.logp(actions))
            old_policy_actions_logp = _make_time_major(
                old_policy_action_dist.logp(actions)
            )
            is_ratio = torch.clamp(
                torch.exp(prev_actions_logp - old_policy_actions_logp), 0.0, 2.0
            )
            logp_ratio = is_ratio * torch.exp(actions_logp - prev_actions_logp)
            self._is_ratio = is_ratio

            advantages = vtrace_returns.pg_advantages.to(logp_ratio.device)
            surrogate_loss = torch.min(
                advantages * logp_ratio,
                advantages
                * torch.clamp(
                    logp_ratio,
                    1 - self.config["clip_param"],
                    1 + self.config["clip_param"],
                ),
            )

            # mean_kl_loss = reduce_mean_valid(action_kl)
            mean_kl_loss = mean_reverse_kl_loss = reduce_mean_valid(reverse_action_kl)
            mean_policy_loss = -reduce_mean_valid(surrogate_loss)

            # The value function loss.
            value_targets = vtrace_returns.vs.to(values_time_major.device)
            delta = values_time_major - value_targets
            mean_vf_loss = 0.5 * reduce_mean_valid(torch.pow(delta, 2.0))

            # The entropy loss.
            mean_entropy = reduce_mean_valid(_make_time_major(action_dist.entropy()))

        else:
            logger.debug("Using PPO surrogate loss (vtrace=False)")

            # Prepare KL for Loss
            _make_time_major(prev_action_dist.kl(action_dist))
            reverse_action_kl = _make_time_major(action_dist.kl(prev_action_dist))
            actions_logp = _make_time_major(action_dist.logp(actions))
            prev_actions_logp = _make_time_major(prev_action_dist.logp(actions))
            logp_ratio = torch.exp(actions_logp - prev_actions_logp)

            advantages = _make_time_major(train_batch[Postprocessing.ADVANTAGES])
            surrogate_loss = torch.min(
                advantages * logp_ratio,
                advantages
                * torch.clamp(
                    logp_ratio,
                    1 - self.config["clip_param"],
                    1 + self.config["clip_param"],
                ),
            )

            # mean_kl_loss = reduce_mean_valid(action_kl)
            mean_kl_loss = mean_reverse_kl_loss = reduce_mean_valid(reverse_action_kl)
            mean_policy_loss = -reduce_mean_valid(surrogate_loss)

            # The value function loss.
            value_targets = _make_time_major(train_batch[Postprocessing.VALUE_TARGETS])
            delta = values_time_major - value_targets
            mean_vf_loss = 0.5 * reduce_mean_valid(torch.pow(delta, 2.0))

            # The entropy loss.
            mean_entropy = reduce_mean_valid(_make_time_major(action_dist.entropy()))

        # The summed weighted loss.
        total_loss = mean_policy_loss - mean_entropy * self.entropy_coeff
        # Optional additional KL Loss
        if self.config["use_kl_loss"]:
            total_loss += self.kl_coeff * mean_reverse_kl_loss

        # Optional vf loss (or in a separate term due to separate
        # optimizers/networks).
        loss_wo_vf = total_loss
        if not self.config["_separate_vf_optimizer"]:
            total_loss += mean_vf_loss * self.config["vf_loss_coeff"]

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = mean_policy_loss
        model.tower_stats["mean_kl_loss"] = mean_kl_loss
        model.tower_stats["mean_reverse_kl_loss"] = mean_reverse_kl_loss
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["value_targets"] = value_targets
        model.tower_stats["vf_explained_var"] = explained_variance(
            torch.reshape(value_targets, [-1]),
            torch.reshape(values_time_major, [-1]),
        )

        # Return one total loss or two losses: vf vs rest (policy + kl).
        if self.config["_separate_vf_optimizer"]:
            return loss_wo_vf, mean_vf_loss
        else:
            return total_loss


class MMDAPPO(APPO):
    def get_default_policy_class(cls, config):
        return MMDTorchPolicy
