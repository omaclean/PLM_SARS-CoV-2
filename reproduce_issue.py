
from Bio import Align

aligner = Align.PairwiseAligner()
aligner.mode = 'local'
seq1 = "ACGT"
seq2 = "ACG"
alignment = aligner.align(seq1, seq2)[0]

print("Type:", type(alignment))
print("Dir:", dir(alignment))

try:
    print("Indices:", alignment.indices)
except AttributeError as e:
    print("Error accessing indices:", e)

try:
    print("Aligned:", alignment.aligned)
except AttributeError as e:
    print("Error accessing aligned:", e)

try:
    print("Coordinates:", alignment.coordinates)
except AttributeError as e:
    print("Error accessing coordinates:", e)
