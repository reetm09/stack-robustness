import os
import pickle

import numpy as np
import pytest

from stack_robustness.genes import load_stack_genes, vocabulary_aliases


def test_load_stack_genes_reads_a_pickled_list_of_numpy_strings(tmp_path):
    path = tmp_path / "genes.pkl"
    path.write_bytes(pickle.dumps([np.str_("A1BG"), np.str_("TP53")], protocol=4))

    assert load_stack_genes(path) == ["A1BG", "TP53"]


HGNC_HEADER = "hgnc_id\tsymbol\tprev_symbol\talias_symbol\tensembl_gene_id\n"


def test_vocabulary_aliases_rename_data_genes_through_their_ensembl_ids(tmp_path):
    hgnc = tmp_path / "hgnc.tsv"
    hgnc.write_text(
        HGNC_HEADER
        + "HGNC:1\tATP5MC3\tATP5G3\t\tENSG01\n"  # renamed since the data were annotated
        + "HGNC:2\tH3-3A\tH3F3A\t\tENSG02\n"  # the vocabulary kept the old name, as the data did
        + "HGNC:3\tNEWNAME\tOLDNAME\tOTHER\tENSG03\n"  # two of its names are in the vocabulary
    )
    vocab = ["ATP5MC3", "H3F3A", "NEWNAME", "OTHER"]
    data_genes = ["ATP5G3", "H3F3A", "OLDNAME", "UNKNOWN"]
    ensembl = ["ENSG01", "ENSG02", "ENSG03", "ENSG99"]

    aliases = vocabulary_aliases(data_genes, ensembl, vocab, hgnc)

    assert aliases == {"ATP5G3": "ATP5MC3"}


class _Exploit:
    def __reduce__(self):
        return (os.system, ("true",))


def test_load_stack_genes_refuses_pickles_that_call_other_code(tmp_path):
    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps([_Exploit()]))

    with pytest.raises(pickle.UnpicklingError):
        load_stack_genes(path)
