import numpy as np
import torch

from verl.trainer.ppo.aspo import apply_aspo_shaping


class MockTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        # Map token id to a single ASCII char; id 96 -> '`', others -> 'a'
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        s = []
        for i in ids:
            if i == 96:
                s.append("`")
            else:
                s.append("a")
        return "".join(s)

    def encode(self, text, add_special_tokens=False):
        # One char per token scheme used in decode
        return [0] * len(text)


def _mk_sample(first_code_at: int | None, T: int) -> torch.Tensor:
    # make a response of length T with optional "```" starting at given index
    x = [1] * T
    if first_code_at is not None and 0 <= first_code_at <= T - 3:
        x[first_code_at : first_code_at + 3] = [96, 96, 96]
    return torch.tensor(x, dtype=torch.long)


def test_apply_aspo_shaping_basics():
    B, T = 6, 12
    # Group ids: two groups of 3
    group_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    responses = torch.stack(
        [
            _mk_sample(2, T),   # G0 elig early
            _mk_sample(6, T),   # G0 elig late
            _mk_sample(None, T),  # G0 no code
            _mk_sample(4, T),   # G1 elig
            _mk_sample(None, T),  # G1 no code
            _mk_sample(8, T),   # G1 elig late
        ],
        dim=0,
    )

    # All tokens valid
    response_mask = torch.ones((B, T), dtype=torch.float32)

    # Token-level rewards: last token 1.0 for some (mark correctness)
    token_level_rewards = torch.zeros((B, T), dtype=torch.float32)
    for i in [0, 1, 3, 5]:
        token_level_rewards[i, -1] = 1.0

    # Base advantages: ones
    advantages = torch.ones((B, T), dtype=torch.float32)

    shaped = apply_aspo_shaping(
        advantages=advantages,
        response_mask=response_mask,
        group_ids=group_ids,
        token_level_rewards=token_level_rewards,
        responses=responses,
        tokenizer=MockTokenizer(),
        non_tensor_batch=None,
        delta=-2.0,
        k=0.7,
        require_code_pass=False,
    )

    # Non-eligible remain unchanged: idx 2 (no code), idx 4 (no code)
    assert torch.allclose(shaped[2], advantages[2])
    assert torch.allclose(shaped[4], advantages[4])

    # Eligible samples changed
    for i in [0, 1, 3, 5]:
        assert not torch.allclose(shaped[i], advantages[i])

    # Bias magnitude per token is bounded by k * |A_seq| = 0.7 * 1.0 = 0.7
    diff = (shaped - advantages) * response_mask
    assert (diff.abs().max().item() <= 0.7000001)

