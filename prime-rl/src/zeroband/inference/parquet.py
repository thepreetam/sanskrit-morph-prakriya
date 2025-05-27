import pyarrow as pa
from typing import TYPE_CHECKING

# Guard vllm imports for dry run
if TYPE_CHECKING:
from vllm import RequestOutput
else:
    # Provide dummy types for runtime if vllm is not there.
    RequestOutput = type('RequestOutput', (), {'request_id': '', 'outputs': [], 'prompt_token_ids': []})

from zeroband.inference.rewards import RequestRewards
from zeroband.utils.parquet import pa_schema


def get_parquet_table(
    request_outputs: list[RequestOutput],
    request_rewards: list[RequestRewards],
    proofs: list[bytes],
    step: int,
    target_lengths: list[int],
) -> pa.Table:
    # Iterator over proofs
    proof_iter = iter(proofs)

    # Create flattened list of records for PyArrow table
    records = []
    for request_output, request_rewards_obj, target_length in zip(request_outputs, request_rewards, target_lengths):
        assert request_output.request_id == request_rewards_obj.request_id
        for output, reward_obj in zip(request_output.outputs, request_rewards_obj.rewards):
            assert output.index == reward_obj.completion_id
            
            prompt_len = len(request_output.prompt_token_ids)
            output_len = len(output.token_ids)
            
            # Construct loss_mask: 0 for prompt, 1 for output
            loss_mask = [0] * prompt_len + [1] * output_len
            
            # Extract logprobs for output tokens
            # output.logprobs is a list of dicts: [{token_id: logprob}, ...]
            # We need a flat list of logprobs for the output tokens.
            # The structure of output.logprobs needs to be handled carefully.
            # Assuming output.logprobs is a list of logprobs corresponding to output.token_ids
            # OR, if it's a dict like vLLM often provides (token_id -> logprob for *all* vocab items at each position),
            # we need to select the logprob of the chosen token_id.
            # For simplicity, assuming output.logprobs ALREADY contains the logprobs of the generated tokens.
            # This might need adjustment based on the exact structure vLLM provides for output.logprobs.
            output_logprobs = []
            if output.logprobs and output_len > 0:
                # Standard vLLM output.logprobs is a list of dicts, one per generated token.
                # Each dict maps token IDs to their logprobs AT THAT STEP.
                # We need the logprob of the token that was actually sampled (output.token_ids[i]).
                try:
                    for i in range(output_len):
                        # output.token_ids[i] gives the ID of the token sampled at step i
                        # output.logprobs[i] is a dict of {token_id: logprob} for all tokens at step i
                        # So, we get the logprob of the sampled token.
                        sampled_token_id = output.token_ids[i]
                        logprob_for_sampled_token = output.logprobs[i].get(sampled_token_id, -float('inf')) # Default to -inf if not found
                        output_logprobs.append(logprob_for_sampled_token)
                except (IndexError, TypeError, AttributeError) as e:
                    # Fallback or error logging if logprobs structure is not as expected
                    # print(f"Warning: Could not extract logprobs for an output. Error: {e}. Setting to empty.")
                    output_logprobs = [-float('inf')] * output_len # Or handle error more gracefully
            elif output_len > 0:
                 output_logprobs = [-float('inf')] * output_len # Placeholder if logprobs are missing

            records.append(
                {
                    "input_tokens": request_output.prompt_token_ids,
                    "output_tokens": output.token_ids,
                    "logprobs": output_logprobs,
                    "loss_mask": loss_mask,
                    "advantages": reward_obj.advantage,
                    "rewards": reward_obj.reward,
                    "task_rewards": reward_obj.task_reward,
                    "length_penalties": reward_obj.length_penalty,
                    "proofs": next(proof_iter) if len(output.token_ids) > 1 else b"",
                    "step": step,
                    "target_lengths": target_length,
                }
            )

    return pa.Table.from_pylist(records, schema=pa_schema)
