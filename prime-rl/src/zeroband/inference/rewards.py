from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import numpy as np
from pydantic_config import BaseConfig

# Guard vllm imports for dry run
if TYPE_CHECKING:
from vllm import CompletionOutput, RequestOutput
else:
    # Provide dummy types for runtime if vllm is not there.
    # These are used by the DummyRequestOutput/DummyCompletionOutput in infer.py for structure.
    CompletionOutput = type('CompletionOutput', (), {'index': 0, 'text': '', 'token_ids': [], 'logprobs': [], 'cumulative_logprob': 0.0, 'finish_reason': 'length'})
    RequestOutput = type('RequestOutput', (), {'request_id': '', 'outputs': []})

from zeroband.inference.genesys import TaskType, get_reward_function
from zeroband.utils.logger import get_logger
# Custom task import for Sanskrit
from zeroband.custom_tasks.sanskrit_morph.verifier import verify_sanskrit_form 

# Global logger
logger = get_logger("INFER")


class LenRewardsConfig(BaseConfig):
    reward_type: Literal["exact", "max", "clip"] = "max"
    target_length_sampling: Literal["discrete", "range"] = "discrete"
    length_prompt_location: Literal["system_prompt", "instruction"] = "system_prompt"

    # applicable if target_length_sampling == "range"
    min_length: int = 1000
    max_length: int = 24000

    # applicable if target_length_sampling == "discrete"
    target_lengths: list[float] = [500, 1000, 2000, 3000]

    # applicable for reward_type max and exact
    reward_coef: float = 0.0003

    # only applicable for reward_type == "max"
    max_reward_delta: float = 0.5


@dataclass
class CompletionReward:
    completion_id: int  # type(CompletionOutput.index)
    reward: float
    task_reward: float
    length_penalty: float
    advantage: float | None = None


@dataclass
class RequestRewards:
    request_id: str  # type(RequestOutput.request_id)
    rewards: list[CompletionReward]


def _compute_completion_reward(
    completion_output: CompletionOutput,
    verification_info: dict,
    task_type: TaskType,
    config: LenRewardsConfig | None,
) -> CompletionReward:
    """
    Computes the reward from a single vLLM completion output given the
    task type (e.g. math, code, etc.) and information on how to verify
    the output. Also supports an optional length penalty.

    Args:
        completion_output: The completion output to compute the reward for.
        verification_info: The verification info for the completion output.
        task_type: The task type for the completion output.
        config: The config for the rewards.

    Returns:
        A CompletionReward object.
    """
    task_reward: float
    reward: float
    length_penalty: float = 0.0

    if task_type == "sanskrit_morph":
        # Assuming completion_output.text is the detokenized generated Sanskrit string
        # And verification_info contains {"metadata": original_metadata_dict_or_str}
        # Ensure your verifier.py is in the specified path and verify_sanskrit_form is importable.
        is_correct = verify_sanskrit_form(verification_info["metadata"], completion_output.text)
        task_reward = float(is_correct) # Convert boolean to 0.0 or 1.0
        reward = task_reward # For Sanskrit, reward is initially the task_reward
        # Length penalty is kept at 0 for sanskrit_morph for now
    else:
        # Existing logic for other task types
        compute_reward_func = get_reward_function(task_type)
        task_reward = compute_reward_func(completion_output.text, verification_info)
        reward = task_reward # Start with task_reward

        # Compute length penalty for non-Sanskrit tasks if applicable
        target_length = verification_info.get("target_length", 0)
        if config and target_length > 0: # Ensure config is not None
        output_length = len(completion_output.token_ids)
        if config.reward_type == "exact":
            length_penalty = abs(target_length - output_length) * config.reward_coef
            reward -= length_penalty
        elif config.reward_type == "max":
                # Ensure reward is non-negative before multiplication
                current_task_reward_for_penalty = max(0, reward) 
            raw_value = config.reward_coef * (target_length - output_length) + config.max_reward_delta
                length_multiplier = max(0, min(1, raw_value))
                # Penalty is how much the reward is reduced by, or the factor it's multiplied by.
                # Let's consider the reduction case for clarity if reward is just task_reward.
                # If reward = task_reward * length_multiplier, then penalty is task_reward * (1-length_multiplier)
                reward = current_task_reward_for_penalty * length_multiplier
                length_penalty = current_task_reward_for_penalty * (1 - length_multiplier) # More explicit penalty value
        elif config.reward_type == "clip":
                if output_length > target_length:
                    length_penalty = reward # The penalty is the full reward if it's clipped to 0
                    reward = 0.0
                else:
                    length_penalty = 0.0 # No penalty if not clipped
        else:
            raise ValueError(f"Invalid reward type: {config.reward_type}")

    return CompletionReward(
        completion_id=completion_output.index, 
        reward=reward, 
        task_reward=task_reward, 
        length_penalty=length_penalty
    )


def _compute_request_rewards(
    request_output: RequestOutput,
    verification_info: dict,
    task_type: TaskType,
    config: LenRewardsConfig | None,
) -> RequestRewards:
    """
    Computes the rewards and advantages from a single vLLM request output given
    the task type (e.g. math, code, etc.) and information on how to verify all
    completions in the request output.

    Args:
        request_output: The request output to compute the rewards for.
        verification_info: The verification info for the request output.
        task_type: The task type for the request output.
        config: The config for the rewards.

    Returns:
        A dictionary containing the rewards, task rewards, and length penalties
        for each completion in the request output.
    """
    completion_rewards: list[CompletionReward] = []
    for output in request_output.outputs:
        args = (output, verification_info, task_type, config)
        completion_rewards.append(_compute_completion_reward(*args))

    # Compute advantage (normalized rewards)
    # Ensure there's at least one reward to avoid division by zero or NaN issues with std if only one sample.
    if completion_rewards:
        reward_array = np.array([cr.reward for cr in completion_rewards], dtype=np.float32)
        if len(reward_array) > 1:
            mean_reward = reward_array.mean()
            std_reward = reward_array.std(ddof=1) # ddof=1 for sample standard deviation
            advantage_array = (reward_array - mean_reward) / (std_reward + 1e-8) # Added epsilon for stability
        else: # Single completion, advantage is typically 0 or reward itself (context dependent)
            advantage_array = np.array([0.0], dtype=np.float32) # Or reward_array, depends on RL formulation
        
        for cr, advantage in zip(completion_rewards, advantage_array):
            cr.advantage = float(advantage)

    return RequestRewards(request_id=request_output.request_id, rewards=completion_rewards)


def compute_rewards(
    request_outputs: list[RequestOutput],
    verification_infos: list[dict],
    task_types: list[TaskType],
    config: LenRewardsConfig | None,
) -> list[RequestRewards]:
    """
    Computes the rewards and advantages for a list of vLLM request outputs
    given their task types and verification infos.

    Args:
        request_outputs: The request outputs to compute the rewards for.
        verification_infos: The verification infos for the request outputs.
        task_types: The task types for the request outputs.
        config: The config for the rewards.

    Returns:
        A list of RequestRewards objects.
    """

    max_workers = min(32, len(request_outputs)) if len(request_outputs) > 0 else 1 # Ensure max_workers is at least 1
    futures = []
    # Use ThreadPoolExecutor only if there are tasks to submit
    if len(request_outputs) > 0:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for request, info, task_type in zip(request_outputs, verification_infos, task_types):
            args = (request, info, task_type, config)
            futures.append(executor.submit(_compute_request_rewards, *args))
    return list(future.result() for future in futures)
    return [] # Return empty list if no request_outputs
