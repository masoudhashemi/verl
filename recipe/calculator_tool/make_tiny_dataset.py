import argparse
import os
from typing import List, Tuple

import datasets


PROBLEMS: List[Tuple[str, str]] = [
    ("Compute 12 + 7 using the calculator tool. Return only the final answer in \\boxed{}.", "19"),
    ("What is (3 + 5) * 4? Use the calculator tool. Final answer in \\boxed{}.", "32"),
    ("Evaluate 100 - 6*7 with the calculator tool. Final in \\boxed{}.", "58"),
    ("Compute 2**5 + 10 using the calculator tool. Final in \\boxed{}.", "42"),
    ("What is 81 // 9 + 3? Use the tool. Final in \\boxed{}.", "12"),
]


def to_row(idx: int, prompt: str, answer: str):
    # Minimal RLHF row; embed tools activation per sample via extra_info.tools_kwargs
    return {
        "data_source": "calculator_tiny",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "MATH",
        "reward_model": {"ground_truth": str(answer)},
        "agent_name": "tool_agent",
        "extra_info": {
            "index": idx,
            # enable tools for this sample; keys must match tool names in the tool registry
            "tools_kwargs": {
                "calculator": {
                    "create_kwargs": {},
                    "execute_kwargs": {},
                }
            },
            "need_tools_kwargs": True,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/calculator_tool", help="Output directory for parquet files")
    parser.add_argument("--train_size", type=int, default=20)
    parser.add_argument("--test_size", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build a small dataset by cycling through PROBLEMS
    train_rows = [to_row(i, *PROBLEMS[i % len(PROBLEMS)]) for i in range(args.train_size)]
    test_rows = [to_row(i, *PROBLEMS[i % len(PROBLEMS)]) for i in range(args.test_size)]

    train = datasets.Dataset.from_list(train_rows)
    test = datasets.Dataset.from_list(test_rows)

    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    train.to_parquet(train_path)
    test.to_parquet(test_path)

    print(f"Wrote: {train_path}\nWrote: {test_path}")


if __name__ == "__main__":
    main()

