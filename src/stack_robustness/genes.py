"""STACK's gene vocabulary (the Hugging Face checkpoint's basecount_1000per_15000max.pkl)."""

import pickle
from collections import Counter

import pandas as pd

# The file is a list of numpy string scalars; nothing else may be constructed while loading it.
_ALLOWED = {
    ("numpy._core.multiarray", "scalar"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy", "dtype"),
}


class _NumpyStringsOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a gene-list pickle")


def load_stack_genes(path) -> list:
    """Gene symbols in STACK's vocabulary order, loaded without running pickled code."""
    with open(path, "rb") as f:
        return [str(gene) for gene in _NumpyStringsOnly(f).load()]


def vocabulary_aliases(data_genes, ensembl_ids, vocab, hgnc_path) -> dict:
    """Data gene symbols to rename so they match STACK's vocabulary.

    Datasets annotated in different years name the same gene differently: Norman et al.
    (2019) has ATP5G3 where STACK's vocabulary has ATP5MC3. A data gene already in the
    vocabulary (matched case-insensitively, as STACK does) keeps its name. Otherwise its
    Ensembl ID is looked up in HGNC's table (hgnc_complete_set.txt). If exactly one of that
    gene's current, previous and alias symbols is in the vocabulary, and no other data gene
    holds or claims it, the gene is renamed to it.
    """
    in_vocab = {g.upper(): g for g in vocab}
    taken = {g.upper() for g in data_genes if g.upper() in in_vocab}
    table = pd.read_csv(
        hgnc_path, sep="\t", dtype=str, usecols=["symbol", "prev_symbol", "alias_symbol", "ensembl_gene_id"]
    )
    names = {}
    for row in table.itertuples(index=False):
        if isinstance(row.ensembl_gene_id, str):
            others = [s for col in (row.prev_symbol, row.alias_symbol) if isinstance(col, str) for s in col.split("|")]
            names[row.ensembl_gene_id] = {s.upper() for s in [row.symbol, *others]}
    aliases = {}
    for gene, ensembl in zip(data_genes, ensembl_ids):
        if gene.upper() in in_vocab:
            continue
        candidates = (names.get(ensembl, set()) & in_vocab.keys()) - taken
        if len(candidates) == 1:
            aliases[gene] = in_vocab[candidates.pop()]
    claims = Counter(aliases.values())
    return {gene: target for gene, target in aliases.items() if claims[target] == 1}
