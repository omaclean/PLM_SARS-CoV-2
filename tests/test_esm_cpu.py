"""
Minimal ESM/Transformer tests that run on CPU.

These tests verify that the PyTorch/ESM/transformer pipeline works correctly
using the smallest available ESM-2 model (esm2_t6_8M_UR50D) in CPU mode.

Marked with @pytest.mark.esm_cpu so they can be run selectively in CI.
"""

import pytest
import torch
import numpy as np

# Skip entire module if transformers is not available
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def esm_model_and_tokenizer():
    """Load smallest ESM-2 model once per test module (CPU-only)."""
    from transformers import AutoTokenizer, EsmForMaskedLM
    
    model_name = "facebook/esm2_t6_8M_UR50D"  # 8M params, 6 layers - smallest ESM-2
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.eval()
    
    device = torch.device("cpu")
    model.to(device)
    
    return model, tokenizer, device


@pytest.fixture
def short_protein_sequence():
    """A short test protein sequence (20 amino acids)."""
    return "MKTIIALSYIFCLVFADYKD"


@pytest.fixture
def very_short_sequence():
    """Very short sequence for fast tests (5 amino acids)."""
    return "MKTII"


class TestESMModelLoading:
    """Test that ESM model loads and runs on CPU."""
    
    @pytest.mark.esm_cpu
    def test_model_loads_successfully(self, esm_model_and_tokenizer):
        """Verify model and tokenizer load without errors."""
        model, tokenizer, device = esm_model_and_tokenizer
        
        assert model is not None
        assert tokenizer is not None
        assert device == torch.device("cpu")
    
    @pytest.mark.esm_cpu
    def test_model_is_in_eval_mode(self, esm_model_and_tokenizer):
        """Verify model is in evaluation mode."""
        model, _, _ = esm_model_and_tokenizer
        assert not model.training


class TestESMTokenization:
    """Test ESM tokenization functionality."""
    
    @pytest.mark.esm_cpu
    def test_tokenize_sequence(self, esm_model_and_tokenizer, short_protein_sequence):
        """Test that protein sequence tokenizes correctly."""
        _, tokenizer, _ = esm_model_and_tokenizer
        
        tokens = tokenizer(short_protein_sequence, return_tensors="pt")
        
        assert "input_ids" in tokens
        assert tokens["input_ids"].shape[0] == 1  # batch size
        # Length should be sequence + special tokens (CLS, EOS)
        assert tokens["input_ids"].shape[1] == len(short_protein_sequence) + 2
    
    @pytest.mark.esm_cpu
    def test_tokenize_batch(self, esm_model_and_tokenizer):
        """Test batch tokenization of multiple sequences."""
        _, tokenizer, _ = esm_model_and_tokenizer
        
        sequences = ["MKTII", "ARNDCQE", "GHILKMFPSTWYV"]
        tokens = tokenizer(sequences, return_tensors="pt", padding=True)
        
        assert tokens["input_ids"].shape[0] == 3  # 3 sequences


class TestESMForwardPass:
    """Test ESM model forward pass and output structure."""
    
    @pytest.mark.esm_cpu
    def test_forward_pass_produces_logits(self, esm_model_and_tokenizer, very_short_sequence):
        """Test that forward pass produces logits of correct shape."""
        model, tokenizer, device = esm_model_and_tokenizer
        
        tokens = tokenizer(very_short_sequence, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = model(**tokens)
        
        # Check logits exist and have correct dimensions
        assert hasattr(outputs, "logits")
        batch_size, seq_len, vocab_size = outputs.logits.shape
        assert batch_size == 1
        assert seq_len == len(very_short_sequence) + 2  # +2 for special tokens
        assert vocab_size == tokenizer.vocab_size
    
    @pytest.mark.esm_cpu
    def test_forward_pass_produces_hidden_states(self, esm_model_and_tokenizer, very_short_sequence):
        """Test that forward pass can return hidden states."""
        model, tokenizer, device = esm_model_and_tokenizer
        
        tokens = tokenizer(very_short_sequence, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        
        assert hasattr(outputs, "hidden_states")
        assert outputs.hidden_states is not None
        # Should have embeddings + n_layers hidden states
        # ESM2-8M has 6 layers, so 7 hidden states total
        assert len(outputs.hidden_states) >= 2
    
    @pytest.mark.esm_cpu
    def test_logits_are_finite(self, esm_model_and_tokenizer, short_protein_sequence):
        """Test that logits contain no NaN or Inf values."""
        model, tokenizer, device = esm_model_and_tokenizer
        
        tokens = tokenizer(short_protein_sequence, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = model(**tokens)
        
        assert torch.isfinite(outputs.logits).all(), "Logits contain NaN or Inf"


class TestESMEmbeddings:
    """Test ESM embedding extraction."""
    
    @pytest.mark.esm_cpu
    def test_extract_mean_embedding(self, esm_model_and_tokenizer, very_short_sequence):
        """Test extraction of mean sequence embedding."""
        model, tokenizer, device = esm_model_and_tokenizer
        
        tokens = tokenizer(very_short_sequence, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        
        # Get last layer hidden states, excluding special tokens
        last_hidden = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
        # Exclude CLS (first) and EOS (last) tokens
        seq_hidden = last_hidden[1:-1]  # [seq_len-2, hidden_dim]
        
        mean_embedding = seq_hidden.mean(dim=0)
        
        assert mean_embedding.shape[0] == model.config.hidden_size
        assert torch.isfinite(mean_embedding).all()
    
    @pytest.mark.esm_cpu
    def test_embedding_differs_for_different_sequences(self, esm_model_and_tokenizer):
        """Test that different sequences produce different embeddings."""
        model, tokenizer, device = esm_model_and_tokenizer
        
        seq1 = "MKTII"
        seq2 = "ARNDQ"
        
        def get_embedding(seq):
            tokens = tokenizer(seq, return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items()}
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
            return outputs.hidden_states[-1][0][1:-1].mean(dim=0)
        
        emb1 = get_embedding(seq1)
        emb2 = get_embedding(seq2)
        
        # Embeddings should be different
        assert not torch.allclose(emb1, emb2, atol=1e-5)


class TestESMLogProbabilities:
    """Test log probability calculations from ESM outputs."""
    
    @pytest.mark.esm_cpu
    def test_log_softmax_on_logits(self, esm_model_and_tokenizer, very_short_sequence):
        """Test applying log softmax to get log probabilities."""
        model, tokenizer, device = esm_model_and_tokenizer
        
        tokens = tokenizer(very_short_sequence, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = model(**tokens)
        
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        
        # Log probs should be <= 0
        assert (log_probs <= 0).all()
        # Exp of log probs should sum to 1 per position
        probs = torch.exp(log_probs)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    @pytest.mark.esm_cpu
    def test_sequence_grammaticality_calculation(self, esm_model_and_tokenizer, very_short_sequence):
        """Test computing sequence 'grammaticality' (sum of log probs)."""
        model, tokenizer, device = esm_model_and_tokenizer
        
        tokens = tokenizer(very_short_sequence, return_tensors="pt")
        input_ids = tokens["input_ids"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        log_probs = torch.nn.functional.log_softmax(outputs.logits[0], dim=-1)
        
        # Calculate grammaticality: sum of log probs for observed tokens
        # Skip special tokens (first and last)
        grammaticality = 0.0
        for i, token_id in enumerate(input_ids[0][1:-1]):
            pos_log_probs = log_probs[i + 1]  # +1 for CLS token offset
            grammaticality += pos_log_probs[token_id].item()
        
        # Grammaticality should be negative (sum of log probs)
        assert grammaticality < 0
        assert np.isfinite(grammaticality)


class TestESMIntegrationWithFunctions:
    """Test integration with Functions_HuggingFace.py functions."""
    
    @pytest.mark.esm_cpu
    def test_embed_sequence_function(self, esm_model_and_tokenizer, very_short_sequence):
        """Test the embed_sequence function from Functions_HuggingFace."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        
        from Functions_HuggingFace import embed_sequence
        
        model, tokenizer, device = esm_model_and_tokenizer
        
        # Create a compatible batch_converter
        def batch_converter(data):
            labels = [d[0] for d in data]
            sequences = [d[1] for d in data]
            tokens = tokenizer(sequences, return_tensors="pt", padding=True)
            return labels, sequences, tokens["input_ids"]
        
        # Create a compatible alphabet mock
        class AlphabetAdapter:
            def __init__(self, tok):
                self.padding_idx = tok.pad_token_id
                self.all_toks = list(tok.get_vocab().keys())
            
            def get_idx(self, token):
                return tokenizer.convert_tokens_to_ids(token)
        
        alphabet = AlphabetAdapter(tokenizer)
        
        # The embed_sequence function expects model with output_hidden_states
        # This should work with HuggingFace transformers
        results, logits, mean_emb, full_emb = embed_sequence(
            very_short_sequence,
            model,
            device,
            model_layers=6,  # Last layer for 6-layer model
            batch_converter=batch_converter,
            alphabet=alphabet
        )
        
        assert logits is not None
        assert mean_emb is not None
        assert full_emb is not None
        assert len(full_emb) == len(very_short_sequence)
    
    @pytest.mark.esm_cpu
    def test_get_sequence_entropy_with_real_logits(self, esm_model_and_tokenizer, very_short_sequence):
        """Test get_sequence_entropy with real model logits."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        
        from Functions_HuggingFace import get_sequence_entropy
        
        model, tokenizer, device = esm_model_and_tokenizer
        
        tokens = tokenizer(very_short_sequence, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = model(**tokens)
        
        # get_sequence_entropy expects logits including BOS/EOS
        logits = outputs.logits[0].cpu()
        
        class AlphabetAdapter:
            def get_idx(self, token):
                return tokenizer.convert_tokens_to_ids(token)
        
        entropy = get_sequence_entropy(very_short_sequence, logits, AlphabetAdapter())
        
        assert len(entropy) == len(very_short_sequence)
        assert all(e >= 0 for e in entropy)  # Entropy is non-negative


# Allow running this file directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "esm_cpu"])
