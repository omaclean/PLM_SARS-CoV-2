
import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock libraries before importing the module
sys.modules['IPython'] = MagicMock()
sys.modules['IPython.display'] = MagicMock()

# Create a mock for Bio that allows submodule access
bio_mock = MagicMock()
sys.modules['Bio'] = bio_mock
sys.modules['Bio.Seq'] = MagicMock()
sys.modules['Bio.SeqIO'] = MagicMock()
sys.modules['Bio.SeqRecord'] = MagicMock()
sys.modules['Bio.PDB'] = MagicMock()
sys.modules['Bio.Align'] = MagicMock()
sys.modules['Bio.SeqUtils'] = MagicMock()
sys.modules['py3Dmol'] = MagicMock()

# Import the function to test
# We need to manually mock the imports inside the file if they are top-level
# But since we mocked sys.modules, it should be fine.
# However, we need to make sure we can import Functions_HuggingFace without errors.

# Let's try to import and if it fails due to other dependencies, we mock them too.
try:
    from Functions_HuggingFace import visualise_mutations_on_pdb
except ImportError:
    # Mock other dependencies
    sys.modules['pandas'] = MagicMock()
    sys.modules['numpy'] = MagicMock()
    sys.modules['torch'] = MagicMock()
    sys.modules['esm'] = MagicMock()
    sys.modules['matplotlib'] = MagicMock()
    sys.modules['matplotlib.pyplot'] = MagicMock()
    sys.modules['seaborn'] = MagicMock()
    sys.modules['scipy'] = MagicMock()
    sys.modules['scipy.special'] = MagicMock()
    
    from Functions_HuggingFace import visualise_mutations_on_pdb

class TestVisualisation(unittest.TestCase):
    
    @patch('Functions_HuggingFace.PDB.PDBParser')
    @patch('Functions_HuggingFace.Align.PairwiseAligner')
    @patch('Functions_HuggingFace.py3Dmol.view')
    @patch('Functions_HuggingFace.display')
    @patch('Functions_HuggingFace.HTML')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data="PDB DATA")
    def test_visualise_mutations(self, mock_open, mock_html, mock_display, mock_view, mock_aligner, mock_parser):
        
    @patch('Functions_HuggingFace.PDB.PDBParser')
    @patch('Functions_HuggingFace.Align.PairwiseAligner')
    @patch('Functions_HuggingFace.py3Dmol.view')
    @patch('Functions_HuggingFace.display')
    @patch('Functions_HuggingFace.HTML')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data="PDB DATA")
    def test_visualise_mutations(self, mock_open, mock_html, mock_display, mock_view, mock_aligner, mock_parser):
        
        # --- Test Case 1: Monomer ---
        print("\nTesting Monomer Case...")
        mock_structure_monomer = MagicMock()
        mock_parser.return_value.get_structure.return_value = mock_structure_monomer
        
        # Mock Monomer (1 chain)
        mock_chain_A = MagicMock()
        mock_chain_A.id = 'A'
        mock_model_monomer = MagicMock()
        mock_model_monomer.get_chains.return_value = [mock_chain_A]
        mock_model_monomer.__iter__.return_value = [mock_chain_A]
        
        mock_structure_monomer.__iter__.return_value = iter([mock_model_monomer])
        
        # Mock Residues
        mock_residue = MagicMock()
        mock_residue.id = (' ', 10, ' ') # Residue 10
        mock_residue.resname = 'ALA'
        mock_chain_A.__iter__.return_value = [mock_residue]
        
        # Mock Alignment
        mock_alignment = MagicMock()
        mock_alignment.score = 100
        mock_alignment.indices = [[0], [0]] 
        mock_aligner.return_value.align.return_value = [mock_alignment]
        
        # Call function
        visualise_mutations_on_pdb('monomer.pdb', 'A', ['A1T'])
        
        # Verify doAssembly=True for monomer
        mock_view.return_value.addModel.assert_called_with("PDB DATA", 'pdb', {'doAssembly': True})
        
        # --- Test Case 2: Multimer (Single Model) ---
        print("\nTesting Multimer (Single Model) Case...")
        mock_structure_multimer = MagicMock()
        mock_parser.return_value.get_structure.return_value = mock_structure_multimer
        
        # Mock Multimer (2 chains in 1 model)
        mock_chain_B = MagicMock()
        mock_chain_B.id = 'B'
        mock_model_multimer = MagicMock()
        mock_model_multimer.get_chains.return_value = [mock_chain_A, mock_chain_B]
        mock_model_multimer.__iter__.return_value = [mock_chain_A, mock_chain_B]
        
        mock_structure_multimer.__iter__.return_value = iter([mock_model_multimer])
        
        # Mock Residues for Chain B
        mock_chain_B.__iter__.return_value = [mock_residue]
        
        # Call function
        visualise_mutations_on_pdb('multimer.pdb', 'A', ['A1T'])
        
        # Verify doAssembly=False for multimer
        mock_view.return_value.addModel.assert_called_with("PDB DATA", 'pdb', {'doAssembly': False})
        
        # --- Test Case 3: Multimer (Multi-Model) ---
        print("\nTesting Multimer (Multi-Model) Case...")
        mock_structure_multimodel = MagicMock()
        mock_parser.return_value.get_structure.return_value = mock_structure_multimodel
        
        # Mock 2 Models
        mock_model_1 = MagicMock()
        mock_model_2 = MagicMock()
        mock_model_1.get_chains.return_value = [mock_chain_A] # Just to satisfy monomer check
        
        # Make structure iterable return 2 models
        mock_structure_multimodel.__iter__.return_value = iter([mock_model_1, mock_model_2])
        
        # Call function
        visualise_mutations_on_pdb('multimodel.pdb', 'A', ['A1T'])
        
        # Verify addModel called multiple times (once for each model)
        # We expect at least 2 calls with string data (not file handle)
        # Note: The exact string value depends on PDBIO mock, which is hard to check here.
        # But we can check call count.
        # Initial call for monomer + initial call for multimer + 2 calls for multimodel = 4 calls total
        self.assertEqual(mock_view.return_value.addModel.call_count, 4)
        
        print("Test passed!")

if __name__ == '__main__':
    unittest.main()
