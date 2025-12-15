import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Selection
from scipy.spatial.distance import cdist

def get_coords(chain, atom_name="CA"):
    """
    Extracts coordinates for a specific atom type from a chain.
    Returns a list of residues and a numpy array of coordinates.
    """
    coords = []
    residue_ids = []
    
    # Filter for standard amino acids to ensure alignment
    for residue in chain:
        if Selection.unfold_entities(residue, 'A'): # Checks for atom presence
            if residue.has_id(atom_name):
                atom = residue[atom_name]
                coords.append(atom.get_coord())
                # Store residue number for reference (Sequence ID)
                residue_ids.append(residue.get_id()[1]) 
                
    return np.array(residue_ids), np.array(coords)

def calculate_matrices(pdb_file, chain_ids=('A', 'B', 'C'), target_chain='A'):
    """
    Calculates 4 distance matrices for a homo-trimer.
    Handles both single-model (chains A,B,C) and multi-model (Model 0,1,2 with chain A) PDBs.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('HA_Trimer', pdb_file)
    models = list(structure)
    
    # Check if we have multiple models (Biological Assembly format)
    if len(models) >= 3:
        print(f"Detected {len(models)} models. Assuming biological assembly (1 protomer per model).")
        print(f"Extracting Chain {target_chain} from Models 0, 1, and 2.")
        
        # Extract coordinates from the same chain in different models
        cA_ids, cA_coords = get_coords(models[0][target_chain])
        cB_ids, cB_coords = get_coords(models[1][target_chain])
        cC_ids, cC_coords = get_coords(models[2][target_chain])
        
    else:
        # Single model format (e.g. Chains A, B, C are the three protomers)
        print("Detected single model. Assuming chains represent protomers.")
        model = structure[0]
        cA_ids, cA_coords = get_coords(model[chain_ids[0]])
        cB_ids, cB_coords = get_coords(model[chain_ids[1]])
        cC_ids, cC_coords = get_coords(model[chain_ids[2]])

    # Check for length consistency
    if not (len(cA_coords) == len(cB_coords) == len(cC_coords)):
        print(f"Warning: Chain lengths differ. A:{len(cA_coords)}, B:{len(cB_coords)}, C:{len(cC_coords)}")
        print("Matrices will be truncated to the minimum common length for safety.")
        min_len = min(len(cA_coords), len(cB_coords), len(cC_coords))
        cA_coords = cA_coords[:min_len]
        cB_coords = cB_coords[:min_len]
        cC_coords = cC_coords[:min_len]
        # Realignment is safer, but truncation serves for a quick check.

    N = len(cA_coords)
    print(f"Processing {N} residues per monomer...")

    # 2. Calculate Distance Matrices using Scipy cdist (Euclidean)
    
    # Matrix 1: Intra (A vs A)
    # Distance between Res_i(A) and Res_j(A)
    dist_intra = cdist(cA_coords, cA_coords, metric='euclidean')

    # Matrix 2: Inter Pair 1 (A vs B)
    # Distance between Res_i(A) and Res_j(B)
    dist_inter_AB = cdist(cA_coords, cB_coords, metric='euclidean')

    # Matrix 3: Inter Pair 2 (A vs C)
    # Distance between Res_i(A) and Res_j(C)
    dist_inter_AC = cdist(cA_coords, cC_coords, metric='euclidean')

    # Matrix 4: Minimum Distance (Epistatic Contact Map)
    # We stack them and take the minimum along the stack axis
    # We must preserve the diagonal in intra (0 distance) for the min operation?
    # Usually for contact maps we want physical interaction. 
    # Self-interaction (i=i) is 0. 
    dist_min = np.minimum(dist_intra, np.minimum(dist_inter_AB, dist_inter_AC))

    return {
        "ids": cA_ids,
        "intra": dist_intra,
        "inter_AB": dist_inter_AB,
        "inter_AC": dist_inter_AC,
        "min_combined": dist_min
    }

def save_matrix(matrix, residue_ids, filename):
    df = pd.DataFrame(matrix, index=residue_ids, columns=residue_ids)
    df.to_csv(filename)
    print(f"Saved: {filename}")

# --- Usage ---
# Replace 'path/to/your/ha_structure.pdb' with your file
# Ensure chain IDs match your PDB (e.g., A, B, C or A, C, E etc.)

try:
    # Example call for 4WE4_assembly.pdb (multi-model structure)
    # We want to analyze Chain A (HA1) across the 3 protomers (Models 0, 1, 2)
    data = calculate_matrices('/home3/oml4h/PLM_SARS-CoV-2/Sequences/4WE4_assembly.pdb', target_chain='A')
    
    # save_matrix(data['intra'], data['ids'], '/home3/oml4h/PLM_SARS-CoV-2/Results/structure_play')
    # save_matrix(data['inter_AB'], data['ids'], 'matrix_2_inter_AB.csv')
    # save_matrix(data['inter_AC'], data['ids'], 'matrix_3_inter_AC.csv')
    save_matrix(data['min_combined'], data['ids'], '/home3/oml4h/PLM_SARS-CoV-2/Results/structure_play/matrix_4_min_combined.csv')
    pass 
except Exception as e:
    print(e)