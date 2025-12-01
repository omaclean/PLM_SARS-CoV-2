
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
        
        # Setup Mocks
        mock_structure = MagicMock()
        mock_parser.return_value.get_structure.return_value = mock_structure
        
        # Mock Monomer (1 chain)
        mock_chain = MagicMock()
        mock_chain.id = 'A'
        mock_model = [mock_chain]
        mock_structure.__iter__.return_value = [mock_model]
        mock_structure.get_chains.return_value = [mock_chain] # Monomer
        
        # Mock Residues
        mock_residue = MagicMock()
        mock_residue.id = (' ', 10, ' ') # Residue 10
        mock_residue.resname = 'ALA'
        mock_chain.__iter__.return_value = [mock_residue]
        
        # Mock Alignment
        mock_alignment = MagicMock()
        mock_alignment.score = 100
        mock_alignment.indices = [[0], [0]] # Match user index 0 to pdb index 0
        mock_aligner.return_value.align.return_value = [mock_alignment]
        
        # Mock View
        mock_view_instance = mock_view.return_value
        
        # Call function
        mutation_list = ['A1T'] # Mutation at index 0 (1-based)
        visualise_mutations_on_pdb('dummy.pdb', 'A', mutation_list)
        
        # Verify View Creation
        mock_view.assert_called_with(width=800, height=600)
        mock_view_instance.addModel.assert_called()
        
        # Verify Styling
        # Should add style for residue 10 with a color
        # Since it's a monomer, selector should NOT have 'chain'
        expected_selector = {'resi': 10}
        
        # Check if addStyle was called with this selector
        calls = mock_view_instance.addStyle.call_args_list
        found_selector = False
        for call in calls:
            args, _ = call
            if args[0] == expected_selector:
                found_selector = True
                break
        
        self.assertTrue(found_selector, "Did not find style application for residue 10 without chain ID (monomer mode)")
        
        # Verify Legend
        mock_display.assert_called()
        mock_html.assert_called()
        
        print("Test passed!")

if __name__ == '__main__':
    unittest.main()
