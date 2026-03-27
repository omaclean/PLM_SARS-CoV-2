import sys
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Mock heavy machine learning libraries
sys.modules['torch'] = MagicMock()
sys.modules['esm'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['safetensors'] = MagicMock()
sys.modules['safetensors.torch'] = MagicMock()

# Append the root directory to sys.path so we can import Functions_HuggingFace
REPO_ROOT = "/home3/oml4h/PLM_SARS-CoV-2"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import the functions from the central library
from Functions_HuggingFace import (
    _resolve_plm_max_nt_length,
    _is_probably_nucleotide_sequence,
    _build_coordinate_map,
    _load_comparison_protein_sequence,
    _save_key_matrix,
    _raw_codon_to_aa_prob,
    _build_aa20_average_and_reconstruction,
    _flattened_fit_metrics,
    validate_mutational_matrix,
    get_ranked_mutations,
    _load_single_focal_reference,
    _ensure_plm_probability_matrix,
)

def test_resolve_plm_max_nt_length():
    # Test protein length limit conversion
    assert _resolve_plm_max_nt_length(max_aa_length=100) == 300
    # Test codon-aware truncation (302 -> 300)
    assert _resolve_plm_max_nt_length(max_nt_length=302) == 300
    # Test both limits (min wins)
    assert _resolve_plm_max_nt_length(max_aa_length=200, max_nt_length=300) == 300
    # Test None
    assert _resolve_plm_max_nt_length() is None

def test_is_probably_nucleotide_sequence():
    assert _is_probably_nucleotide_sequence("ATGC") is True
    assert _is_probably_nucleotide_sequence("AUGC") is True
    assert _is_probably_nucleotide_sequence("atgc") is True
    assert _is_probably_nucleotide_sequence("MRKLP") is False
    assert _is_probably_nucleotide_sequence("ATGCN-") is True

def test_build_coordinate_map():
    query = "ACGT"
    target = "AC-GT" # Gap in query/target alignment
    # Wait, our coordinate map uses global alignment
    # If target has a gap relative to query, query indices map to target indices
    # Simple case: identities
    mapping = _build_coordinate_map("ACGT", "ACGT")
    assert mapping == {0: 0, 1: 1, 2: 2, 3: 3}
    
    # Case with indel
    mapping = _build_coordinate_map("ACT", "ACGT")
    # A->A (0:0), C->C (1:1), T->T (2:3)
    assert mapping == {0: 0, 1: 1, 2: 3}

def test_raw_codon_to_aa_prob():
    # Mock data
    codon_mutation_df = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4]], 
        index=["AAA", "AAG"], 
        columns=["AAA", "AAG"]
    )
    aa_to_codons_all = {"K": ["AAA", "AAG"]}
    
    # AAA -> K should sum AAA->AAA + AAA->AAG = 0.1 + 0.2 = 0.3
    prob = _raw_codon_to_aa_prob("AAA", "K", codon_mutation_df, aa_to_codons_all)
    assert np.isclose(prob, 0.3)

def test_flattened_fit_metrics():
    obs = pd.DataFrame([[1, 2], [3, 4]])
    pred = pd.DataFrame([[1.1, 1.9], [3.2, 3.8]])
    metrics = _flattened_fit_metrics(obs, pred)
    assert metrics["n_entries"] == 4
    assert metrics["corr"] > 0.9
    assert metrics["rmse"] < 0.2

def test_get_ranked_mutations():
    # 2 positions, 3 AAs (A, C, G)
    prob_matrix = pd.DataFrame(
        [[0.1, 0.8], [0.2, 0.1], [0.7, 0.1]],
        index=["A", "C", "G"],
        columns=["pos0", "pos1"]
    )
    ref_seq = "AA" # Pos 0 is A, Pos 1 is A
    obs_muts = [(1, "C")] # Pos 1 changed to C
    
    ranked_df, obs_df = get_ranked_mutations(prob_matrix, ref_seq, obs_muts)
    
    # Pos 0: A->C (0.2), A->G (0.7)
    # Pos 1: A->C (0.1), A->G (0.1)
    # Total candidates: 4
    assert len(ranked_df) == 4
    # Highest prob should be Pos 0: A->G (0.7)
    assert ranked_df.iloc[0]["AA"] == "G"
    assert ranked_df.iloc[0]["Position"] == 0
    
    # Observed should be Pos 1: A->C (0.1)
    assert len(obs_df) == 1
    assert obs_df.iloc[0]["Probability"] == 0.1
    assert obs_df.iloc[0]["AA"] == "C"

def test_load_single_focal_reference():
    # Create a dummy fasta
    dummy_fasta = "/tmp/dummy_focal.fasta"
    with open(dummy_fasta, "w") as f:
        f.write(">test_seq\nATGGTGTAA\n")
    
    # Mock _translate_nt_to_protein because it depends on CodonTable
    with patch("Functions_HuggingFace._translate_nt_to_protein", return_value="MV"):
        res = _load_single_focal_reference(dummy_fasta, "lineage_A")
        assert res["header"] == "test_seq"
        assert res["lineage"] == "lineage_A"
        assert res["nucleotide"] == "ATGGTGTAA"
        assert res["protein"] == "MV"


    def test_reference_sequence_validity():
        # Create a dummy fasta with nucleotide length multiple of 3
        dummy_fasta = "/tmp/dummy_valid_ref.fasta"
        with open(dummy_fasta, "w") as f:
            f.write(">valid_ref\nATGGCC\n")

        res = _load_single_focal_reference(dummy_fasta, "lineage_test")

        # Nucleotide length must be multiple of 3
        assert len(res["nucleotide"]) % 3 == 0

        # Protein translation must not contain unknowns ('X') or alignment gaps ('-')
        assert "X" not in res["protein"]
        assert "-" not in res["protein"]


    def test_lineage_sequences_are_detected_as_nucleotide(tmp_path):
        # Create a dummy diversity fasta with sequences that are nucleotide-like
        fasta = tmp_path / "diversity_nuc_like.fasta"
        fasta.write_text(">s1\nATGATGATG\n>s2\nATGCCC\n")

        records = list(SeqIO.parse(str(fasta), "fasta"))
        assert len(records) == 2
        # Should detect nucleotide-like content
        assert any(_is_probably_nucleotide_sequence(str(rec.seq)) for rec in records)


    def test_lineage_sequences_not_detected_for_protein():
        # Create protein-like records and ensure detection is False
        prot_recs = [SeqRecord(Seq("ACDEFGHIK"), id="p1"), SeqRecord(Seq("LMNPQRSTV"), id="p2")]
        assert not any(_is_probably_nucleotide_sequence(str(rec.seq)) for rec in prot_recs)

def test_ensure_plm_probability_matrix_cache():
    # Test that it uses cache
    cache = {("model1", "seq1"): {"mutation_matrix": [[0.5]], "amino_acids": ["A"], "positions": ["0"], "sequence": "A"}}
    output_path = "/tmp/test_matrix.csv"
    
    with patch("Functions_HuggingFace._write_plm_probability_matrix") as mock_write:
        _ensure_plm_probability_matrix(
            "seq1", output_path, "model1", None, "base_model", 33, force_recompute=False, cache=cache
        )
        # Should call write since it's in cache
        mock_write.assert_called_once()

