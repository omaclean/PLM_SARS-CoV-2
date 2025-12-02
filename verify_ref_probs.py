import sys
from unittest.mock import MagicMock

# Mock esm and other dependencies
sys.modules['esm'] = MagicMock()
# Mock Bio and submodules
mock_bio = MagicMock()
sys.modules['Bio'] = mock_bio
sys.modules['Bio.PDB'] = MagicMock()
sys.modules['Bio.Align'] = MagicMock()
sys.modules['Bio.SeqUtils'] = MagicMock()
sys.modules['Bio.Seq'] = MagicMock()
sys.modules['Bio.SeqRecord'] = MagicMock()
sys.modules['Bio.SeqIO'] = MagicMock()

sys.modules['IPython.display'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['seaborn'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

import torch
import numpy as np

# Now import the function
from Functions_HuggingFace import get_reference_probabilities

def test_get_reference_probabilities():
    # Mock sequence
    sequence = "AC"
    seq_len = len(sequence)
    vocab_size = 5 # A, C, D, E, F
    
    # Mock alphabet
    mock_alphabet = MagicMock()
    # Map A->0, C->1
    mock_alphabet.get_idx.side_effect = lambda x: {'A': 0, 'C': 1}.get(x, 0)
    
    # Mock logits: [cls] + sequence + [eos] -> seq_len + 2
    # Shape: (4, 5)
    # Let's make logits such that probabilities are predictable
    # Pos 1 (A): High logit for index 0
    # Pos 2 (C): High logit for index 1
    logits = torch.zeros(4, 5)
    logits[1, 0] = 10.0 # A at pos 1
    logits[2, 1] = 10.0 # C at pos 2
    
    # Call function
    probs = get_reference_probabilities(sequence, logits, mock_alphabet)
    
    print(f"Sequence: {sequence}")
    print(f"Probabilities: {probs}")
    
    assert len(probs) == seq_len, "Probabilities list length should match sequence length"
    assert all(isinstance(x, float) for x in probs), "Probabilities should be floats"
    assert probs[0] > 0.9, "Probability for A should be high"
    assert probs[1] > 0.9, "Probability for C should be high"
    
    print("Test passed!")

if __name__ == "__main__":
    test_get_reference_probabilities()
