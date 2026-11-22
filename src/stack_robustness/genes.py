"""STACK's gene vocabulary (the Hugging Face checkpoint's basecount_1000per_15000max.pkl)."""

import pickle

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
