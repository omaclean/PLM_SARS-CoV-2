"""
Comprehensive pytest tests for Functions_HuggingFace.py

This test suite covers all functions in the Functions_HuggingFace module,
including mutation functions, translation functions, embedding functions,
entropy functions, scoring functions, and PDB visualization utilities.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO, Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
import tempfile
import os
import bz2
import pickle
import sys

# Add parent directory to path to import Functions_HuggingFace
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Functions_HuggingFace import (
    compressed_pickle, decompress_pickle,
    makeOrfTable, makeMatProteinTable,
    mutate_sequence, DMS, DMS_Table,
    iterative_translate,
    get_sequence_entropy, get_reference_probabilities,
    get_mutation_prob_matrix,
    grammaticality_and_evolutionary_index, get_sequence_grammaticality,
    semantic_calc,
    read_sequences_to_dict, get_mutations, get_indel_mutations,
    get_reference_mutations, revert_sequence,
    format_logits, check_valid,
    align_sequences, create_h3_numbering_map
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_sequence():
    """Provide a sample protein sequence for testing."""
    return "MKTIIALSYIFCLVFA"  # 16 amino acids


@pytest.fixture
def sample_genbank_record():
    """Create a mock GenBank record with CDS and mat_peptide features."""
    record = SeqRecord(
        Seq("ATGGGTAAACCCGGG" * 20),  # 300 nucleotides
        id="test_record"
    )
    
    # Add CDS feature
    cds_feature = SeqFeature(
        FeatureLocation(0, 100),
        type="CDS",
        qualifiers={'gene': ['testgene']}
    )
    record.features.append(cds_feature)
    
    # Add mat_peptide feature
    mat_feature = SeqFeature(
        FeatureLocation(0, 50),
        type="mat_peptide",
        qualifiers={'product': ['test_protein'], 'gene': ['testgene']}
    )
    record.features.append(mat_feature)
    
    return record


@pytest.fixture
def sample_logits():
    """Create sample logits tensor for testing."""
    # Create logits for a 5 amino acid sequence + BOS/EOS tokens
    vocab_size = 33  # ESM alphabet size
    seq_len = 7  # 5 AAs + 2 special tokens
    logits = torch.randn(seq_len, vocab_size)
    return logits


@pytest.fixture
def mock_alphabet():
    """Create a mock alphabet object for testing."""
    class MockAlphabet:
        def __init__(self):
            self.all_toks = ['<cls>', '<pad>', '<eos>', '<unk>'] + list("ARNDCQEGHILKMFPSTWYV")
            self.padding_idx = 1
            
        def get_idx(self, token):
            return self.all_toks.index(token)
    
    return MockAlphabet()


@pytest.fixture
def temp_fasta_file():
    """Create a temporary FASTA file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(">seq1\n")
        f.write("ARNDCQEGHILKMFPSTWYV\n")
        f.write(">seq2\n")
        f.write("ARNDCQEGHILKMFPSTW-Y\n")
        f.write(">seq3_with_mutation\n")
        f.write("AKNDCQEGHILKMFPSTWYV\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ============================================================================
# Compression Functions Tests
# ============================================================================

class TestCompressionFunctions:
    """Test compression and decompression utilities."""
    
    def test_compressed_pickle_and_decompress(self):
        """Test that data can be pickled and unpickled correctly."""
        test_data = {'key': 'value', 'numbers': [1, 2, 3], 'array': np.array([1, 2, 3])}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test')
            
            # Compress and save
            compressed_pickle(filepath, test_data)
            
            # Verify file was created
            assert os.path.exists(filepath + '.pbz2')
            
            # Decompress and load
            loaded_data = decompress_pickle(filepath + '.pbz2')
            
            # Verify data integrity
            assert loaded_data['key'] == test_data['key']
            assert loaded_data['numbers'] == test_data['numbers']
            np.testing.assert_array_equal(loaded_data['array'], test_data['array'])
    
    def test_compressed_pickle_complex_objects(self):
        """Test compression of complex nested structures."""
        complex_data = {
            'dataframe': pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}),
            'nested': {'level1': {'level2': [1, 2, 3]}}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'complex')
            compressed_pickle(filepath, complex_data)
            loaded = decompress_pickle(filepath + '.pbz2')
            
            pd.testing.assert_frame_equal(loaded['dataframe'], complex_data['dataframe'])
            assert loaded['nested'] == complex_data['nested']


# ============================================================================
# GenBank Annotation Functions Tests
# ============================================================================

class TestGenbankFunctions:
    """Test GenBank annotation parsing functions."""
    
    def test_makeOrfTable(self, sample_genbank_record):
        """Test extraction of CDS features into a dataframe."""
        orf_table = makeOrfTable(sample_genbank_record)
        
        assert isinstance(orf_table, pd.DataFrame)
        assert 'testgene' in orf_table.index
        assert 'Start' in orf_table.columns
        assert 'End' in orf_table.columns
        assert 'Part' in orf_table.columns
        assert 'Locations' in orf_table.columns
        assert orf_table.loc['testgene', 'Start'] == 0
        assert orf_table.loc['testgene', 'End'] == 100
    
    def test_makeMatProteinTable(self, sample_genbank_record):
        """Test extraction of mature peptide features."""
        protein_table = makeMatProteinTable(sample_genbank_record)
        
        assert isinstance(protein_table, pd.DataFrame)
        assert 'test_protein' in protein_table.index
        assert 'ORF' in protein_table.columns
        assert protein_table.loc['test_protein', 'ORF'] == 'testgene'
        assert protein_table.loc['test_protein', 'Start'] == 0
        assert protein_table.loc['test_protein', 'End'] == 50


# ============================================================================
# Mutation Functions Tests
# ============================================================================

class TestMutationFunctions:
    """Test mutation generation and application functions."""
    
    def test_mutate_sequence_single_mutation(self):
        """Test applying a single point mutation."""
        reference = "ARNDCQE"
        mutations = ["A1K"]  # Change position 1 from A to K
        result = mutate_sequence(reference, mutations)
        assert result == "KRNDCQE"
    
    def test_mutate_sequence_multiple_mutations(self):
        """Test applying multiple mutations."""
        reference = "ARNDCQE"
        mutations = ["A1K", "R2G", "E7D"]
        result = mutate_sequence(reference, mutations)
        assert result == "KGNDCQD"
    
    def test_mutate_sequence_no_mutations(self):
        """Test that sequence is unchanged with empty mutations list."""
        reference = "ARNDCQE"
        result = mutate_sequence(reference, [])
        assert result == reference
    
    def test_DMS_generates_all_mutants(self):
        """Test that DMS generates all single amino acid substitutions."""
        reference = "AR"  # 2 amino acids
        mutants = DMS(reference)
        
        # Should generate 2 positions * 19 other amino acids = 38 mutants
        # (excluding the reference amino acid at each position)
        assert len(mutants) == 38
        
        # Verify they are SeqRecord objects
        assert all(isinstance(m, SeqRecord) for m in mutants)
        
        # Check a specific mutation
        mutation_ids = [m.id for m in mutants]
        assert "A1R" in mutation_ids  # Position 1 A->R
    
    def test_DMS_Table_returns_dataframe(self):
        """Test that DMS_Table returns proper dataframe."""
        reference = "AR"
        df = DMS_Table(reference)
        
        assert isinstance(df, pd.DataFrame)
        assert 'Mutations' in df.columns
        assert 'Sequence' in df.columns
        assert len(df) == 38  # 2 * 19 mutations


# ============================================================================
# Translation Functions Tests
# ============================================================================

class TestTranslationFunctions:
    """Test nucleotide to protein translation functions."""
    
    def test_iterative_translate_basic(self):
        """Test basic translation of nucleotide sequence."""
        # ATG = M, GGT = G, AAA = K, CCC = P
        nucleotide = "ATGGGTAAACCC"
        protein = iterative_translate(nucleotide)
        assert protein == "MGKP"
    
    def test_iterative_translate_with_gaps(self):
        """Test translation handles gap characters."""
        nucleotide = "ATG---GGGTAA"
        protein = iterative_translate(nucleotide)
        assert protein == "M-G"  # Gap codons become '-'
    
    def test_iterative_translate_truncate_at_stop(self):
        """Test truncation at stop codon."""
        # ATG = M, GGT = G, TAA = stop
        nucleotide = "ATGGGTTAAAAA"
        protein = iterative_translate(nucleotide, truncate_proteins=True)
        assert protein == "MG"  # Should stop before stop codon
    
    def test_iterative_translate_incomplete_codon(self):
        """Test handling of incomplete codons."""
        nucleotide = "ATGGG"  # Only 5 nucleotides
        protein = iterative_translate(nucleotide)
        assert protein == "M"  # Should only translate complete codons


# ============================================================================
# Entropy and Probability Functions Tests
# ============================================================================

class TestEntropyFunctions:
    """Test entropy and probability calculation functions."""
    
    def test_get_sequence_entropy_shape(self, sample_logits, mock_alphabet):
        """Test that entropy calculation returns correct shape."""
        sequence = "ARNDC"
        entropies = get_sequence_entropy(sequence, sample_logits, mock_alphabet)
        
        assert isinstance(entropies, list)
        assert len(entropies) == len(sequence)
        assert all(isinstance(e, float) for e in entropies)
    
    def test_get_sequence_entropy_positive_values(self, sample_logits, mock_alphabet):
        """Test that entropy values are non-negative."""
        sequence = "ARNDC"
        entropies = get_sequence_entropy(sequence, sample_logits, mock_alphabet)
        assert all(e >= 0 for e in entropies)
    
    def test_get_reference_probabilities_shape(self, sample_logits, mock_alphabet):
        """Test reference probability calculation returns correct shape."""
        sequence = "ARNDC"
        probs = get_reference_probabilities(sequence, sample_logits, mock_alphabet)
        
        assert isinstance(probs, list)
        assert len(probs) == len(sequence)
        assert all(isinstance(p, float) for p in probs)
    
    def test_get_reference_probabilities_range(self, sample_logits, mock_alphabet):
        """Test that probabilities are in valid range [0, 1]."""
        sequence = "ARNDC"
        probs = get_reference_probabilities(sequence, sample_logits, mock_alphabet)
        assert all(0 <= p <= 1 for p in probs)


# ============================================================================
# Scoring Functions Tests
# ============================================================================

class TestScoringFunctions:
    """Test mutation scoring and grammaticality functions."""
    
    def test_get_sequence_grammaticality(self, sample_logits, mock_alphabet):
        """Test sequence grammaticality calculation."""
        sequence = "ARNDC"
        score = get_sequence_grammaticality(sequence, sample_logits, mock_alphabet)
        
        assert isinstance(score, (float, np.floating))
    
    def test_get_sequence_grammaticality_with_masking(self, sample_logits, mock_alphabet):
        """Test grammaticality with position masking."""
        sequence = "ARNDC"
        mask_pos = [1, 2]  # Mask positions 1 and 2
        score = get_sequence_grammaticality(sequence, sample_logits, mock_alphabet, mask_pos=mask_pos)
        
        assert isinstance(score, (float, np.floating))
    
    def test_semantic_calc_identical_sequences(self):
        """Test semantic distance between identical embeddings is zero."""
        embedding1 = [1.0, 2.0, 3.0, 4.0]
        embedding2 = [1.0, 2.0, 3.0, 4.0]
        
        distance = semantic_calc(embedding1, embedding2)
        assert distance == 0.0
    
    def test_semantic_calc_different_sequences(self):
        """Test semantic distance between different embeddings."""
        embedding1 = [1.0, 2.0, 3.0]
        embedding2 = [2.0, 3.0, 4.0]
        
        distance = semantic_calc(embedding1, embedding2)
        assert distance > 0
        assert distance == 3.0  # Sum of absolute differences
    
    def test_grammaticality_and_evolutionary_index_no_mutations(self, mock_alphabet):
        """Test with no mutations returns zeros."""
        word_pos_prob = {}
        seq = "ARNDC"
        mutations = []
        
        gm, ev = grammaticality_and_evolutionary_index(word_pos_prob, seq, mutations)
        assert gm == 0
        assert ev == 0
    
    def test_check_valid_within_range(self):
        """Test check_valid returns None when value is within range."""
        result = check_valid(5, 0, 10)
        assert result is None
    
    def test_check_valid_outside_range(self):
        """Test check_valid returns value when outside range."""
        result = check_valid(15, 0, 10)
        assert result == 15
        
        result = check_valid(-5, 0, 10)
        assert result == -5


# ============================================================================
# Sequence Reading and Mutation Detection Tests
# ============================================================================

class TestSequenceFunctions:
    """Test sequence I/O and mutation detection functions."""
    
    def test_read_sequences_to_dict(self, temp_fasta_file):
        """Test reading FASTA file into dictionary."""
        sequences = read_sequences_to_dict(temp_fasta_file)
        
        assert isinstance(sequences, dict)
        assert 'seq1' in sequences
        assert 'seq2' in sequences
        assert sequences['seq1'] == "ARNDCQEGHILKMFPSTWYV"
    
    def test_get_mutations_single_mutation(self):
        """Test detection of single point mutation."""
        seq1 = "ARNDCQE"
        seq2 = "AKNDCQE"  # R2K mutation
        
        mutations = get_mutations(seq1, seq2)
        assert len(mutations) == 1
        assert "R2K" in mutations
    
    def test_get_mutations_multiple_mutations(self):
        """Test detection of multiple mutations."""
        seq1 = "ARNDCQE"
        seq2 = "AKNDCGE"  # R2K and Q6G
        
        mutations = get_mutations(seq1, seq2)
        assert len(mutations) == 2
        assert "R2K" in mutations
        assert "Q6G" in mutations
    
    def test_get_mutations_with_deletion(self):
        """Test detection of deletions."""
        seq1 = "ARNDCQE"
        seq2 = "ARND-QE"  # C5 deleted
        
        mutations = get_mutations(seq1, seq2)
        assert len(mutations) == 1
        assert "C5del" in mutations
    
    def test_get_mutations_no_changes(self):
        """Test that identical sequences return no mutations."""
        seq1 = "ARNDCQE"
        seq2 = "ARNDCQE"
        
        mutations = get_mutations(seq1, seq2)
        assert len(mutations) == 0
    
    def test_get_indel_mutations(self):
        """Test detection of insertions and deletions."""
        aligned_ref = "ARND-CQE"
        indel_seq = "ARN-KCQE"
        
        mutations, insertions, deletions = get_indel_mutations(aligned_ref, indel_seq)
        
        assert "D4K" in mutations  # D->K substitution
        assert "ins4K" in insertions  # Insertion at position 4
        assert "del4" in deletions  # Deletion at position 4
    
    def test_get_reference_mutations(self):
        """Test getting mutations between aligned sequences."""
        ref = "ARND-CQE"
        mut = "AKNDKCQE"
        
        mutations = get_reference_mutations(ref, mut)
        assert "R2K" in mutations
    
    def test_revert_sequence(self):
        """Test reverting mutations in a sequence."""
        mutated = "AKNDCQE"
        mutations = ["R2K"]  # This tells us position 2 was originally R
        
        reverted = revert_sequence(mutated, mutations)
        assert reverted == "ARNDCQE"


# ============================================================================
# Alignment Functions Tests
# ============================================================================

class TestAlignmentFunctions:
    """Test sequence alignment utilities."""
    
    def test_align_sequences_local(self):
        """Test local sequence alignment."""
        ref_seq = "ARNDCQEGHILKMFPSTWYV"
        query_seq = "ARNDCQEGHILKMFPSTWYV"
        
        alignment = align_sequences(ref_seq, query_seq, mode='local')
        
        assert alignment is not None
        assert hasattr(alignment, 'score')
        assert alignment.score > 0
    
    def test_align_sequences_global(self):
        """Test global sequence alignment."""
        ref_seq = "ARND"
        query_seq = "AKND"
        
        alignment = align_sequences(ref_seq, query_seq, mode='global')
        
        assert alignment is not None
        assert hasattr(alignment, 'score')
    
    def test_align_sequences_with_gaps(self):
        """Test alignment with gaps."""
        ref_seq = "ARNDCQE"
        query_seq = "ARND-QE"
        
        alignment = align_sequences(ref_seq, query_seq, mode='global')
        assert alignment is not None
    
    def test_create_h3_numbering_map(self, temp_fasta_file):
        """Test H3 numbering map creation from FASTA file."""
        h3_map = create_h3_numbering_map(
            temp_fasta_file,
            reference_id='seq1',
            query_id='seq2',
            signal_peptide_length=5
        )
        
        assert isinstance(h3_map, dict)
        assert len(h3_map) > 0
        
        # Check that mapping contains expected types
        for key, value in h3_map.items():
            assert isinstance(key, int)  # Position should be integer
            assert isinstance(value, str)  # Label should be string


# ============================================================================
# Format and Utility Functions Tests
# ============================================================================

class TestUtilityFunctions:
    """Test miscellaneous utility functions."""
    
    def test_format_logits(self, sample_logits, mock_alphabet):
        """Test logits formatting to dataframe."""
        formatted = format_logits(sample_logits, mock_alphabet)
        
        assert isinstance(formatted, pd.DataFrame)
        # Should have 20 amino acid columns
        assert len(formatted.columns) == 20
        # Should remove BOS and EOS tokens (7 - 2 = 5 rows)
        assert len(formatted) == 5


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_mutation_pipeline(self):
        """Test complete mutation detection and application pipeline."""
        # Original sequence
        ref_seq = "ARNDCQEGHILK"
        
        # Create a mutant
        mutations = ["A1K", "R2G"]
        mutant_seq = mutate_sequence(ref_seq, mutations)
        
        # Detect mutations
        detected = get_mutations(ref_seq, mutant_seq)
        
        # Verify mutations were detected
        assert len(detected) == 2
        assert "A1K" in detected
        assert "R2G" in detected
        
        # Revert mutations
        reverted = revert_sequence(mutant_seq, mutations)
        assert reverted == ref_seq
    
    def test_fasta_read_and_mutate(self, temp_fasta_file):
        """Test reading sequences and detecting mutations."""
        sequences = read_sequences_to_dict(temp_fasta_file)
        
        # Compare seq1 and seq3
        mutations = get_mutations(sequences['seq1'], sequences['seq3_with_mutation'])
        
        assert len(mutations) > 0
        assert any('R2K' in m for m in mutations)  # R->K at position 2


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_mutate_sequence_empty_mutations(self):
        """Test mutation with empty list."""
        seq = "ARNDCQE"
        result = mutate_sequence(seq, [])
        assert result == seq
    
    def test_get_mutations_different_lengths(self):
        """Test mutation detection with sequences of same length."""
        seq1 = "ARND"
        seq2 = "AKND"
        mutations = get_mutations(seq1, seq2)
        assert len(mutations) == 1
    
    def test_iterative_translate_empty_sequence(self):
        """Test translation of empty sequence."""
        result = iterative_translate("")
        assert result == ""
    
    def test_DMS_single_position(self):
        """Test DMS on single amino acid."""
        mutants = DMS("A")
        # Should generate 19 mutants (all other amino acids)
        assert len(mutants) == 19
    
    def test_semantic_calc_with_arrays(self):
        """Test semantic calculation with numpy arrays."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([2.0, 3.0, 4.0])
        
        distance = semantic_calc(arr1, arr2)
        assert distance == 3.0


# ============================================================================
# Pytest Configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
