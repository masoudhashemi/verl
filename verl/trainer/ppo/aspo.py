import re
from typing import Iterable, Optional, Tuple

import numpy as np
import torch


def _batch_decode_prefix(responses: torch.Tensor, response_mask: torch.Tensor, tokenizer) -> list[str]:
    """Decode only the valid response portion per sample.

    Args:
        responses: (B, T_resp) token ids for responses.
        response_mask: (B, T_resp) 1 for valid response tokens, 0 after EOS or tool observation tokens.
        tokenizer: tokenizer with batch_decode/encode methods.

    Returns:
        List[str]: decoded response strings per sample.
    """
    with torch.no_grad():
        lengths = response_mask.sum(dim=-1).to(torch.int64)  # (B,)
        texts: list[str] = []
        for i in range(responses.size(0)):
            L = int(lengths[i].item())
            if L <= 0:
                texts.append("")
                continue
            toks = responses[i, :L].tolist()
            texts.append(tokenizer.decode(toks, skip_special_tokens=True))
        return texts


_DEFAULT_PATTERNS = [
    re.compile(r"```[a-zA-Z0-9_]*"),  # fenced code block start, optional language
    re.compile(r"<\s*code\b", re.IGNORECASE),  # <code ...>
    re.compile(r"<\s*pre\b", re.IGNORECASE),  # <pre ...>
    re.compile(r"\"tool_calls\"\s*:\s*\["),  # OpenAI-style tool_calls
    re.compile(r"\"function_call\"\s*:\s*\{"),  # legacy function_call
    re.compile(r"\"tool_name\"\s*:\s*\""),  # tool calling snippets
]


def _find_first_code_char_index(text: str, patterns: Iterable[re.Pattern]) -> Optional[int]:
    idxs = [m.start() for pat in patterns if (m := pat.search(text))]
    return None if len(idxs) == 0 else min(idxs)


def _approx_char_to_token_index(prefix_text: str, tokenizer) -> int:
    # Use tokenizer to get the approximate number of tokens for the prefix.
    # add_special_tokens=False to align with raw response tokens
    try:
        ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    except TypeError:
        # Some tokenizers require different kwargs
        ids = tokenizer.encode(prefix_text)
        # Try best-effort to drop specials if present
    return len(ids)


def detect_used_code_and_first_pos(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
    non_tensor_batch: Optional[dict] = None,
    prefer_structured_tool_calls: bool = True,
    markdown_backticks: bool = True,
    html_code_tags: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Detect whether each response used code/tool and the token index of the first invocation.

    Returns:
        used_code_mask: (B,) bool tensor
        first_code_pos: (B,) int tensor where -1 means not found
    """
    B = responses.size(0)

    # Build pattern list based on config
    patterns: list[re.Pattern] = []
    if markdown_backticks:
        patterns.append(_DEFAULT_PATTERNS[0])
    if html_code_tags:
        patterns.extend(_DEFAULT_PATTERNS[1:3])
    # JSON tool-calling markers (always useful if present in text)
    patterns.extend(_DEFAULT_PATTERNS[3:])

    texts = _batch_decode_prefix(responses, response_mask, tokenizer)

    used = torch.zeros(B, dtype=torch.bool)
    p_first = torch.full((B,), -1, dtype=torch.int64)

    # If structured messages are available, they indicate tool usage; still map to token via text search
    msgs = None
    if prefer_structured_tool_calls and non_tensor_batch is not None:
        msgs = non_tensor_batch.get("messages")

    for i in range(B):
        t = texts[i] or ""

        # Opportunistic: if messages show tool usage, mark used, but still try to locate in text
        if msgs is not None:
            try:
                # messages is an array of dict with key 'messages'
                msg_pkg = msgs[i]
                # handle numpy object element or dict
                message_list = msg_pkg["messages"] if isinstance(msg_pkg, dict) else getattr(msg_pkg, "messages", None)
                if isinstance(message_list, list):
                    # If any assistant turn contains tool_calls or role switches to tool, consider as code usage
                    for m in message_list:
                        if isinstance(m, dict):
                            if m.get("tool_calls") or m.get("role") == "tool":
                                used[i] = True
                                break
            except Exception:
                # ignore parsing failures; fall back to text detection
                pass

        # Find in text using patterns
        char_idx = _find_first_code_char_index(t, patterns)
        if char_idx is not None:
            used[i] = True
            # Map char index to token index
            prefix = t[:char_idx]
            p_first[i] = _approx_char_to_token_index(prefix, tokenizer)

    return used, p_first


@torch.no_grad()
def apply_aspo_shaping(
    advantages: torch.Tensor,  # (B, T)
    response_mask: torch.Tensor,  # (B, T)
    group_ids: np.ndarray,  # (B,)
    token_level_rewards: torch.Tensor,  # (B, T)
    responses: torch.Tensor,  # (B, T)
    tokenizer,
    non_tensor_batch: Optional[dict] = None,
    *,
    delta: float = -2.0,
    k: float = 0.7,
    require_code_pass: bool = False,
    prefer_structured_tool_calls: bool = True,
    markdown_backticks: bool = True,
    html_code_tags: bool = True,
    code_pass_mask: Optional[torch.Tensor] = None,  # (B,) optional
) -> torch.Tensor:
    """Apply Advantage Shaping Policy Optimization (ASPO) to token-level advantages.

    The shaping adds a bounded bias per eligible sample and broadcasts it over response tokens.
    """
    B, T = advantages.shape
    device = advantages.device

    # Sequence-level base advantage per sample
    resp_len = response_mask.sum(dim=-1).clamp_min(1.0)  # avoid div by zero
    a_seq = (advantages * response_mask).sum(dim=-1) / resp_len  # (B,)

    # Correctness from rewards: sum of token rewards > 0 treated as correct
    seq_reward = (token_level_rewards * response_mask).sum(dim=-1)  # (B,)
    correct_mask = seq_reward > 0.0

    used_code_mask, first_code_pos = detect_used_code_and_first_pos(
        responses=responses,
        response_mask=response_mask,
        tokenizer=tokenizer,
        non_tensor_batch=non_tensor_batch,
        prefer_structured_tool_calls=prefer_structured_tool_calls,
        markdown_backticks=markdown_backticks,
        html_code_tags=html_code_tags,
    )
    eligible = correct_mask & used_code_mask
    if require_code_pass and code_pass_mask is not None:
        eligible = eligible & code_pass_mask.to(eligible.device)

    if eligible.sum() == 0:
        return advantages  # no change

    # Prepare tensors
    first_code_pos = first_code_pos.to(device=device, dtype=torch.float32)
    resp_len_tokens = response_mask.sum(dim=-1).to(device=device, dtype=torch.float32)

    # Broadcast bias per sample within group
    shaped_adv = advantages.clone()
    unique_groups = np.unique(group_ids)
    for g in unique_groups:
        idx_np = group_ids == g
        if not np.any(idx_np):
            continue
        idx = torch.from_numpy(idx_np).to(device)
        elig = idx & eligible.to(device)
        if elig.sum() == 0:
            continue

        p = first_code_pos[elig]
        L = resp_len_tokens[elig]
        mean_p = p.mean()
        mean_L = L.mean()
        if mean_L <= 0:
            continue

        # raw bias for eligible items in this group
        raw_bias = delta * (first_code_pos[idx & eligible.to(device)] - mean_p) / mean_L  # (n_elig,)

        a_base = a_seq[idx & eligible.to(device)]  # (n_elig,)
        # defensive: bound by ±k * |A|
        upper = k * a_base.abs()
        lower = -upper
        bias = torch.maximum(torch.minimum(raw_bias, upper), lower)

        # add bias to all tokens of eligible samples
        # broadcast: (n_elig, 1)
        bias_broadcast = bias.view(-1, 1)
        # gather indices of eligible samples for assignment
        elig_indices = torch.nonzero(idx & eligible.to(device), as_tuple=False).squeeze(-1)
        shaped_adv[elig_indices] = advantages[elig_indices] + bias_broadcast * response_mask[elig_indices]

    return shaped_adv

