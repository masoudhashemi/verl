from verl.utils.reward_score import math_dapo


def compute_score(data_source, solution_str, ground_truth, extra_info):
    """Compute correctness based on final boxed answer.

    Uses the same helper as other math examples. Returns a dict matching VERL's expected format.
    """
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)
    if result.get("pred") is None:
        result["pred"] = ""
    return result

