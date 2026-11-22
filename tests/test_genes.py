import os
import pickle

import numpy as np
import pytest

from stack_robustness.genes import load_stack_genes


def test_load_stack_genes_reads_a_pickled_list_of_numpy_strings(tmp_path):
    path = tmp_path / "genes.pkl"
    path.write_bytes(pickle.dumps([np.str_("A1BG"), np.str_("TP53")], protocol=4))

    assert load_stack_genes(path) == ["A1BG", "TP53"]


class _Exploit:
    def __reduce__(self):
        return (os.system, ("true",))


def test_load_stack_genes_refuses_pickles_that_call_other_code(tmp_path):
    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps([_Exploit()]))

    with pytest.raises(pickle.UnpicklingError):
        load_stack_genes(path)
