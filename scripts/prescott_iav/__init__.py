"""PRESCOTT/ESCOTT diversity-prediction pipeline for influenza A H3N2 HA.

Stage 1 of the pipeline (``prepare_inputs``, ``jet_surrogate``, ``run_escott``)
runs under the PRESCOTT conda env and must stay free of torch/esm, because
``Functions_HuggingFace`` imports both at module scope.  Stage 2
(``scripts/run_prescott_diversity.py``) runs under ``plm_entropy`` and reuses
``run_mutational_accessibility`` verbatim.  The only contract between the two
halves is ``scores/<lineage_key>_<variant>_score_matrix.csv``.
"""
