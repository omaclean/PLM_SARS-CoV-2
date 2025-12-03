import torch
import numpy as np
from Functions_HuggingFace import get_sequence_entropy

def test_get_sequence_entropy():
    # Mock sequence
    sequence = "ACDEF"
    seq_len = len(sequence)
    vocab_size = 20
    
    # Mock logits: [cls] + sequence + [eos] -> seq_len + 2
    # Shape: (seq_len + 2, vocab_size)
    logits = torch.randn(seq_len + 2, vocab_size)
    
    # Mock alphabet (not used in current implementation but required by signature)
    alphabet = None
    
    # Call function
    entropy = get_sequence_entropy(sequence, logits, alphabet)
    
    print(f"Sequence length: {seq_len}")
    print(f"Entropy length: {len(entropy)}")
    print(f"Entropy values: {entropy}")
    
    assert len(entropy) == seq_len, "Entropy list length should match sequence length"
    assert all(isinstance(x, float) for x in entropy), "Entropy values should be floats"
    assert all(x >= 0 for x in entropy), "Entropy should be non-negative"
    
    print("Test passed!")

if __name__ == "__main__":
    test_get_sequence_entropy()
