import sys
from unittest.mock import MagicMock

# Mock esm and other dependencies
sys.modules['esm'] = MagicMock()
sys.modules['Bio'] = MagicMock()
sys.modules['Bio.PDB'] = MagicMock()
sys.modules['Bio.Align'] = MagicMock()
sys.modules['Bio.SeqUtils'] = MagicMock()
sys.modules['IPython.display'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['seaborn'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

import torch
import numpy as np

# Now import the function
# We need to make sure Functions_HuggingFace can be imported even if it imports these things
# The mocks above should handle it.

from Functions_HuggingFace import get_sequence_entropy

def test_get_sequence_entropy():
    # Mock sequence
    sequence = "ACDEF"
    seq_len = len(sequence)
    vocab_size = 20
    
    # Mock logits: [cls] + sequence + [eos] -> seq_len + 2
    # Shape: (seq_len + 2, vocab_size)
    logits = torch.randn(seq_len + 2, vocab_size)
    
    # Mock alphabet
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
