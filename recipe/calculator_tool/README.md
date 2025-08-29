# Calculator Tool + ASPO Example

This recipe demonstrates multi-turn rollouts with an external tool (a simple calculator) incorporated into generation, plus ASPO (Advantage Shaping Policy Optimization) to encourage earlier tool usage during PPO.

What’s included:

- A safe in-process calculator tool.
- A tiny tool-augmented dataset generator.
- A boxed-answer reward function.
- A runnable PPO script enabling multi-turn tool-calls and ASPO.

## Requirements

- GPU recommended for rollout with vLLM backend.
- A chat-instruct model that supports function/tool-calling patterns (Qwen/Qwen2.5-3B-Instruct in the script).
- Python deps from the main VERL environment. No extra packages are required beyond `datasets` (already used in the repo).

## Files

- `calculator_tool.py`: Implements `CalculatorTool` (`BaseTool` subclass) with an OpenAI-style function schema `calculator(expression: string)`.
- `calculator_tool_config.yaml`: Tool registry for the rollout worker. The tool name is `calculator` (must match when enabling per-sample).
- `make_tiny_dataset.py`: Creates a small train/test dataset with prompts, ground-truth, and `extra_info.tools_kwargs` to activate the `calculator` tool per sample.
- `reward.py`: Computes correctness using final boxed answer.
- `run_qwen2.5_3b_ppo.sh`: PPO script enabling multi-turn, tool-calling, and ASPO.

## File-by-file Details

- `calculator_tool.py`
  - Purpose: Defines the calculator tool by subclassing `verl.tools.base_tool.BaseTool`. Exposes an OpenAI-style function schema so the rollout worker can route tool calls.
  - Requirements: Importable by Python (ensure repo root on `PYTHONPATH`); relies on VERL tool interfaces; no external services required.
  - Used by: SGLang rollout worker after loading `tool_config_path`.
  - When: During rollout tool phases:
    - Create: optional per-trajectory state via `create(...)`.
    - Execute: invoked when model emits `calculator(...)` tool call; returns `ToolResponse`.
    - Release: optional cleanup at trajectory end.

- `calculator_tool_config.yaml`
  - Purpose: Registers the tool class and schema for the rollout worker.
  - Requirements: `class_name` must resolve to `recipe.calculator_tool.calculator_tool.CalculatorTool`. The `function.name` must be `calculator` to match `tools_kwargs`.
  - Used by: Rollout worker when `actor_rollout_ref.rollout.multi_turn.tool_config_path` points to this file.
  - When: Loaded at worker initialization; schema is attached per-request when the sample’s `tools_kwargs` activates it.

- `make_tiny_dataset.py`
  - Purpose: Generates small parquet datasets with prompts and ground-truth answers; embeds `extra_info.tools_kwargs` to activate the tool per sample.
  - Requirements: `datasets` library; write permission to `--output_dir`.
  - Used by: You (offline) to produce `train.parquet` and `test.parquet`; then the trainer’s data pipeline reads them.
  - When: Before training; during training, `tools_kwargs` flows through `verl.utils.dataset.rl_dataset` to the rollout worker.

- `reward.py`
  - Purpose: Provides `compute_score` to evaluate correctness based on final boxed answer using `math_dapo` helpers.
  - Requirements: VERL’s reward utilities available; referenced via `custom_reward_function.path/name` in the run script.
  - Used by: PPO trainer post-rollout to compute rewards.
  - When: After generation for each batch during train/val steps.

- `run_qwen2.5_3b_ppo.sh`
  - Purpose: Launches PPO training with multi-turn rollouts, tool-calling, and ASPO.
  - Requirements: GPU recommended; access to the specified models; vLLM installed per VERL docs; dataset paths exist.
  - Used by: You to start the experiment.
  - When: At startup; config enables `data.return_raw_chat=True`, multi-turn + tools, and `algorithm.aspo.*`.

## How It Works

- Tool wiring: The rollout worker loads tools from `actor_rollout_ref.rollout.multi_turn.tool_config_path`. When the model emits a function call to `calculator`, the worker executes it and appends a tool message with the result before continuing generation.
- Per-sample activation: Dataset rows embed `extra_info.tools_kwargs` keyed by the tool name (`calculator`). This injects the tool schema for that request. Set `data.return_raw_chat=True` so the worker has structured messages.
- ASPO: Enabled in PPO via `algorithm.aspo.enabled=true`. It detects tool/code usage (from structured tool-calls or text) and adds a clipped bias to policy advantages of eligible samples (correct + used tool), encouraging earlier tool invocation. Value targets and rewards remain unchanged.

## Quick Start

1) Generate dataset:
   - `python3 recipe/calculator_tool/make_tiny_dataset.py --output_dir data/calculator_tool`
   - Creates `data/calculator_tool/train.parquet` and `test.parquet`.

2) Run PPO with tool-calling + ASPO:
   - `bash recipe/calculator_tool/run_qwen2.5_3b_ppo.sh`
   - You can override model paths via env vars:
     - `ACTOR_MODEL_PATH=... CRITIC_MODEL_PATH=... bash recipe/calculator_tool/run_qwen2.5_3b_ppo.sh`

## Verifying It’s Working

- Tool usage:
  - Logs show transitions into tool-calling; look for `tool_calls` in printed samples or tool-role messages. You should see calculator outputs echoed as tool responses inside the conversation.
  - Ensure the tool schema name `calculator` matches `extra_info.tools_kwargs` in the dataset.
- ASPO applied:
  - Confirm `algorithm.aspo.enabled=true` in the echoed config.
  - ASPO only affects policy advantages; it does not change rewards or critic targets. Over time, eligible samples (correct + used tool) are biased toward earlier tool invocation.

## Customization

- Model/backend:
  - Swap `actor_rollout_ref.model.path` to your model. Keep `data.return_raw_chat=True` and multi-turn enabled.
  - The script uses vLLM rollout (`actor_rollout_ref.rollout.name=vllm`). Adjust TP/BSZ to your hardware.
- Dataset size:
  - `make_tiny_dataset.py --train_size N --test_size M` to scale up.
- Tool behavior:
  - Edit `calculator_tool.py` to support more operations or formatting.
- ASPO tuning:
  - Adjust `algorithm.aspo.delta` and `algorithm.aspo.k`. Detection knobs under `algorithm.aspo.code_detection.*` control structured vs. text markers.

## Troubleshooting

- Model never calls the tool:
  - Use an instruction-tuned model with tool-calling capability; keep prompts explicit (“Use the calculator tool”).
  - Ensure `tool_config_path` is correct and `data.return_raw_chat=True`.
  - Check that dataset rows include `extra_info.tools_kwargs` with a `calculator` key.
- Length issues:
  - Reduce `max_prompt_length`/`max_response_length` or batch sizes. Ensure prompts fit within `max_model_len`.
- CPU-only environment:
  - vLLM typically expects a GPU. Consider smaller models or alternative rollout backends if needed.

For more on ASPO internals, see `recipe/aspo/README.md` and `verl/trainer/ppo/aspo.py`.
