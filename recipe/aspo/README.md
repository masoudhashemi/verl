# Recipe: Advantage Shaping Policy Optimization (ASPO)

This recipe demonstrates how to enable and tune ASPO in VERL. ASPO adds a small, clipped bias to the policy advantages of eligible samples (correct + used code/tool) to encourage earlier code invocation, without modifying rewards or critic targets.

Key design points:

- Advantage shaping only: operates after your normal advantage computation and before actor loss.
- Eligibility: response is correct and used at least one code/tool call (optionally require success).
- Bounded bias: per-sample bias is clipped to ±k·|base_advantage| to keep correctness dominant.
- Group-relative: earlier code than the group mean gets a positive nudge; later gets a negative one.

References: Section 3.3 & 4.5 of the ASPO paper (Eq. (2)).

## How to enable

ASPO is configured under `algorithm.aspo` in the PPO/GRPO trainer config and is off by default.

Minimal overrides to enable with GRPO (group-normalized advantages):

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.aspo.enabled=true \
  algorithm.aspo.delta=-2.0 \
  algorithm.aspo.k=0.7 \
  algorithm.aspo.require_code_pass=false
```

For PPO/GAE, ASPO also works; it will compute a sequence-level base advantage per sample and broadcast the bias over response tokens for the policy loss.

## Detection of code usage

By default, the implementation detects code/tool usage in two ways:

- Structured messages (if available from multi-turn/tooling rollouts): uses `non_tensor_batch["messages"]`.
- Text patterns: looks for fenced blocks (```), `<code>`/`<pre>` tags, and common tool-calling JSON markers.

You can control detection toggles via:

```yaml
algorithm:
  aspo:
    code_detection:
      prefer_structured_tool_calls: true
      markdown_backticks: true
      html_code_tags: true
```

If you have execution telemetry, you can gate eligibility on success:

```yaml
algorithm:
  aspo:
    require_code_pass: true
```

Provide a boolean `code_pass` array in `non_tensor_batch` per sample to activate this gate.

## Recommended hyperparameters

- Conservative: `delta = -2.0`, `k = 0.7`
- Aggressive: `delta = -2.5`, `k = 0.9`

Notes:

- `delta < 0` makes earlier code get a positive bias.
- `k` caps the bias fraction; higher values emphasize shaping more, but should remain < 1.0.

## Sanity checks

Track during training/eval:

- Mean first code position (token index)
- Code usage ratio
- Response length (tokens)
- Optional: code pass ratio, code rounds/lines

Expected behavior: earlier `p_first`, higher code usage, stable correctness.

## Example run: GRPO on GSM8K with ASPO

See `run_aspo_grpo_gsm8k.sh` for a runnable example based on Qwen2 (adjust batch sizes and hardware to your setup).

```bash
export RAY_ADDRESS="http://${RAY_IP:-localhost}:8265"
export WORKING_DIR="${PWD}"
export RUNTIME_ENV="recipe/aspo/runtime_env.yaml"
bash recipe/aspo/run_aspo_grpo_gsm8k.sh
```

## Safety and invariants

- ASPO never changes rewards, returns, or value loss targets; it only modifies the advantage used by the policy loss.
- If a group has no eligible samples or length stats are degenerate, shaping is skipped for that group.
- Shaping is applied only on response tokens (masked by `response_mask`).
