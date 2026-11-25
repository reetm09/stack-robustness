"""The robustness conditions (proposal, Robustness) and the cells each one feeds the models.

Draws are paired across conditions: for a given draw, plate and drug, every condition starts
from the same random ordering of prompt, control-prompt and query cells, and smaller sets are
prefixes of larger ones. A condition then subsamples, contaminates or corrupts those cells.
"""

from dataclasses import dataclass, field

import numpy as np

from .pbmc import DMSO, QUERY_KINDS, Cells

# STACK's 512-cell window at T = 1 (prompt_ratio 0.25, context_ratio 0.4) holds 333 prompt and
# 179 query cells. Two windows per call: up to 666 distinct prompt cells, 179 query cells per kind.
PROMPT_CELLS = 666
QUERY_CELLS = 179


@dataclass(frozen=True)
class Condition:
    name: str
    arm: str  # clean, prompt_size, query_size, thinning, ambient, contamination, plate
    severity: float = 0.0  # x-axis within the arm (cells, rate or fraction)
    target: str = "none"  # what is corrupted: prompt, query, both
    n_prompt: int = PROMPT_CELLS
    n_query: int = QUERY_CELLS  # per query cell type
    thin_rate: float = 0.0
    ambient_rate: float = 0.0
    contamination: float = 0.0  # share of the drug prompt replaced by another drug's T/NK cells
    prompt_plate: str = "same"  # "same" or "other" (plate swap)
    drugs: tuple = field(default=())  # restrict to these drugs (empty: all)
    draws: int = 2

    def corrupts(self, role: str) -> bool:
        return self.target in (role, "both") and (self.thin_rate > 0 or self.ambient_rate > 0)

    @property
    def control_key(self) -> str:
        """Conditions with the same key give identical control-prompt generations for a draw."""
        if self.contamination > 0:
            return "clean"
        return self.name


BOTH_PLATES = ("Belinostat", "Dabrafenib")


def default_conditions(draws: int = 2, clean_draws: int = 3, plate_draws: int = 5) -> list:
    out = [Condition("clean", "clean", draws=clean_draws)]
    for n in (16, 32, 64, 128):
        out.append(Condition(f"prompt_n{n}", "prompt_size", n, n_prompt=n, draws=draws))
    for n in (16, 32, 64, 128):
        out.append(Condition(f"query_n{n}", "query_size", n, n_query=n, draws=draws))
    for target in ("prompt", "query", "both"):
        for rho in (0.10, 0.25, 0.50):
            out.append(Condition(f"thin_{target}_{int(rho * 100)}", "thinning", rho, target, thin_rate=rho, draws=draws))
    for target in ("prompt", "query"):
        for alpha in (0.10, 0.20):
            out.append(Condition(f"ambient_{target}_{int(alpha * 100)}", "ambient", alpha, target,
                                 ambient_rate=alpha, draws=draws))
    for frac in (0.10, 0.25, 0.50):
        out.append(Condition(f"contam_{int(frac * 100)}", "contamination", frac, contamination=frac, draws=draws))
    for plate in ("same", "other"):
        out.append(Condition(f"plate_{plate}", "plate", float(plate == "other"), prompt_plate=plate,
                             drugs=BOTH_PLATES, draws=plate_draws))
    return out


def _order(rows, seed) -> np.ndarray:
    return np.random.default_rng(seed).permutation(rows)


def plates(cells: Cells) -> list:
    return sorted(set(cells.plate))


def drugs_on(cells: Cells, plate: str, cond: Condition, targets) -> list:
    """Drugs with at least one scored target on this plate."""
    found = sorted({t.drug for t in targets if t.plate == plate})
    return [d for d in found if not cond.drugs or d in cond.drugs]


@dataclass
class Inputs:
    """Row indices into Cells for one condition, draw and plate (before count corruption)."""

    query: np.ndarray  # B cells then myeloid cells
    query_kind: np.ndarray
    control_prompt: np.ndarray
    prompts: dict  # drug -> rows


def sample_inputs(cells: Cells, cond: Condition, draw: int, plate: str, drugs, seed: int = 0) -> Inputs:
    p = plates(cells).index(plate)
    prompt_plate = plate if cond.prompt_plate == "same" else next(q for q in plates(cells) if q != plate)
    pp = plates(cells).index(prompt_plate)
    query, kinds = [], []
    for k, kind in enumerate(QUERY_KINDS):
        rows = _order(cells.rows(DMSO, kind, plate), [seed, draw, p, 100 + k])[: cond.n_query]
        query.append(rows)
        kinds += [kind] * len(rows)
    control = _order(cells.rows(DMSO, "T", prompt_plate), [seed, draw, pp, 200])[: cond.n_prompt]
    prompts = {}
    all_drugs = sorted(set(cells.drug) - {DMSO})
    for drug in drugs:
        d = all_drugs.index(drug)
        rows = _order(cells.rows(drug, "T", prompt_plate), [seed, draw, pp, 300 + d])[: cond.n_prompt]
        if cond.contamination > 0:
            others = np.flatnonzero((cells.kind == "T") & (cells.plate == prompt_plate)
                                    & (cells.drug != drug) & (cells.drug != DMSO))
            rng = np.random.default_rng([seed, draw, pp, 400 + d])
            n_bad = int(round(cond.contamination * len(rows)))
            rows = rows.copy()
            rows[len(rows) - n_bad:] = rng.choice(others, size=n_bad, replace=False)
        prompts[drug] = rows
    return Inputs(np.concatenate(query), np.array(kinds), control, prompts)
