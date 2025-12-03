# Tests for Functions_HuggingFace.py

This directory contains comprehensive pytest tests for the `Functions_HuggingFace.py` module.

## Test Coverage

The test suite covers:

### ✅ Compression Functions
- `compressed_pickle()` - Serialization and compression
- `decompress_pickle()` - Decompression and deserialization

### ✅ GenBank Annotation Functions
- `makeOrfTable()` - CDS feature extraction
- `makeMatProteinTable()` - Mature peptide extraction

### ✅ Mutation Functions
- `mutate_sequence()` - Applying point mutations
- `DMS()` - Deep mutational scanning
- `DMS_Table()` - DMS results as dataframe

### ✅ Translation Functions
- `iterative_translate()` - Nucleotide to protein translation
- Handling gaps and incomplete codons
- Stop codon truncation

### ✅ Entropy and Probability Functions
- `get_sequence_entropy()` - Shannon entropy calculation
- `get_reference_probabilities()` - Reference amino acid probabilities

### ✅ Scoring Functions
- `grammaticality_and_evolutionary_index()` - Mutation scoring
- `get_sequence_grammaticality()` - Sequence log-probability
- `semantic_calc()` - L1 distance between embeddings
- `check_valid()` - Range validation

### ✅ Sequence I/O and Mutation Detection
- `read_sequences_to_dict()` - FASTA file parsing
- `get_mutations()` - Point mutation detection
- `get_indel_mutations()` - Insertion/deletion detection
- `get_reference_mutations()` - Aligned sequence mutations
- `revert_sequence()` - Mutation reversal

### ✅ Alignment Functions
- `align_sequences()` - Pairwise sequence alignment
- `create_h3_numbering_map()` - H3 canonical numbering

### ✅ Utility Functions
- `format_logits()` - Logits to dataframe conversion

### ⚠️ Functions Not Tested (Require Model/GPU)
The following functions require trained ESM models and are not tested in the automated suite:
- `embed_sequence()`
- `process_protein_sequence()`
- `embed_protein_sequences()`
- `get_mutation_prob_matrix()`
- `process_sequence_genbank()`
- `process_and_dms_sequence_genbank()`
- `process_fasta()`
- `visualise_mutations_on_pdb()`

## Running Tests

### Run all tests
```bash
pytest tests/test_functions_huggingface.py
```

### Run with verbose output
```bash
pytest tests/test_functions_huggingface.py -v
```

### Run specific test class
```bash
pytest tests/test_functions_huggingface.py::TestMutationFunctions -v
```

### Run specific test
```bash
pytest tests/test_functions_huggingface.py::TestMutationFunctions::test_mutate_sequence_single_mutation -v
```

### Run with coverage report
```bash
pytest tests/test_functions_huggingface.py --cov=Functions_HuggingFace --cov-report=html
```

### Run tests in parallel (faster)
```bash
pytest tests/test_functions_huggingface.py -n auto
```

## Test Organization

Tests are organized into classes by functionality:
- `TestCompressionFunctions` - Pickle compression/decompression
- `TestGenbankFunctions` - GenBank annotation parsing
- `TestMutationFunctions` - Mutation generation and application
- `TestTranslationFunctions` - Nucleotide translation
- `TestEntropyFunctions` - Entropy and probability calculations
- `TestScoringFunctions` - Mutation scoring and grammaticality
- `TestSequenceFunctions` - Sequence I/O and mutation detection
- `TestAlignmentFunctions` - Sequence alignment utilities
- `TestUtilityFunctions` - Miscellaneous utilities
- `TestIntegration` - Multi-function integration tests
- `TestEdgeCases` - Edge cases and error handling

## Continuous Integration

Tests run automatically on every push to `main` or `develop` branches via GitHub Actions. The CI pipeline:

1. **Tests** - Runs on Python 3.8, 3.9, 3.10, and 3.11
2. **Coverage** - Generates coverage reports and uploads to Codecov
3. **Linting** - Checks code style with flake8 and black

See `.github/workflows/tests.yml` for the complete CI configuration.

## Test Fixtures

Common test data is provided via pytest fixtures:
- `sample_sequence` - Example protein sequence
- `sample_genbank_record` - Mock GenBank record
- `sample_logits` - Example model logits
- `mock_alphabet` - Mock ESM alphabet object
- `temp_fasta_file` - Temporary FASTA file for testing

## Requirements

```bash
pip install pytest pytest-cov
pip install biopython pandas numpy torch scipy matplotlib seaborn
```

## Adding New Tests

When adding new functions to `Functions_HuggingFace.py`:

1. Add corresponding tests to `test_functions_huggingface.py`
2. Follow the existing test structure and naming conventions
3. Include docstrings describing what each test validates
4. Test both normal operation and edge cases
5. Run tests locally before committing

## Coverage Goals

Target: **80%+ test coverage** for non-model-dependent functions

Current coverage can be viewed by running:
```bash
pytest tests/test_functions_huggingface.py --cov=Functions_HuggingFace --cov-report=term-missing
```
