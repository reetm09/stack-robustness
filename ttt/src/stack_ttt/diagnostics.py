"""How much a prediction follows its prompt: does replacing a control prompt with A's cells
move the predicted profile along A's own effect?"""

import numpy as np


def prompt_use(profiles: dict, ctrl: np.ndarray, arm: str, steps: int = 1, top: int = 200) -> list:
    """For each prompt order of one held-out pair A+B, the prompt-induced shift (generated with
    the prompt minus generated with a control prompt, as diff scoring uses it) against the
    prompt's own observed effect, on the prompt's `top` genes by absolute effect:
    `r` is their correlation, `relative_size` the mean absolute shift over the mean absolute effect.

    `profiles` is one pair's saved profiles (norman_ttt.py's pairs/<double>.npz): `query_one` is
    B's observed profile, and each combined profile is the median of the two orders, which for
    two values is their mean, so the B->A order is recovered as twice the combined minus A->B.
    """
    obs_b = profiles["query_one"]
    obs_a = 2 * profiles["query"] - obs_b
    diff_ab = profiles[f"{arm}_T{steps}_diff_one"]
    diff_ba = 2 * profiles[f"{arm}_T{steps}_diff"] - diff_ab
    out = []
    for order, shift, effect in (("A->B", diff_ab - obs_b, obs_a - ctrl), ("B->A", diff_ba - obs_a, obs_b - ctrl)):
        idx = np.argsort(-np.abs(effect))[:top]
        s, e = shift[idx], effect[idx]
        r = float(np.corrcoef(s, e)[0, 1]) if s.std() > 0 and e.std() > 0 else float("nan")
        out.append({"prompt_order": order, "r": r, "relative_size": float(np.abs(s).mean() / np.abs(e).mean())})
    return out
