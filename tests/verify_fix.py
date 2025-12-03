
import sys
from Bio import Align

# Mock class for old Biopython alignment without indices
class OldAlignmentMock:
    def __init__(self, aligned):
        self.aligned = aligned
        self.score = 100

    @property
    def indices(self):
        raise AttributeError("'OldAlignmentMock' object has no attribute 'indices'")

# Mock class for new Biopython alignment with indices
class NewAlignmentMock:
    def __init__(self, indices):
        self.indices = indices
        self.score = 100

def test_logic(alignment):
    print(f"Testing with {type(alignment).__name__}...")
    try:
        user_indices = alignment.indices[0]
        pdb_indices = alignment.indices[1]
        print("Accessed .indices directly.")
    except AttributeError:
        print("Caught AttributeError, using fallback...")
        user_indices = []
        pdb_indices = []
        for (u_start, u_end), (p_start, p_end) in zip(*alignment.aligned):
            user_indices.extend(range(u_start, u_end))
            pdb_indices.extend(range(p_start, p_end))
    
    print(f"User indices: {user_indices}")
    print(f"PDB indices: {pdb_indices}")
    return user_indices, pdb_indices

# Test Case 1: Old Biopython behavior
# Aligned: 0-3 matches 0-3
aligned_data = ([(0, 3)], [(0, 3)]) 
old_align = OldAlignmentMock(aligned_data)
u_old, p_old = test_logic(old_align)

# Test Case 2: New Biopython behavior
# Indices: [0, 1, 2], [0, 1, 2]
indices_data = [[0, 1, 2], [0, 1, 2]]
new_align = NewAlignmentMock(indices_data)
u_new, p_new = test_logic(new_align)

# Assertions
assert u_old == [0, 1, 2]
assert p_old == [0, 1, 2]
assert u_new == [0, 1, 2]
assert p_new == [0, 1, 2]

print("\nVerification Successful!")
