from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, EsmConfig


REPO_ROOT = Path("/home3/oml4h/PLM_SARS-CoV-2")
PLANT_CODE_DIR = Path("/home3/oml4h/hugging_face_downloads/PLANT_model/code")
LOCAL_HF_DIR = Path("/home3/oml4h/hugging_face_downloads/PLANT_model")

if str(PLANT_CODE_DIR / "src") not in sys.path:
    sys.path.append(str(PLANT_CODE_DIR / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from plant import TextDataset, embed_sequences, semanticESM, set_encoders, tokenize_sequences
from Functions_HuggingFace import plant_alignment_coverage_metrics, plant_sequence_identity, plant_trim_to_target_length


SUBFOLDER = "variants/PLANT_fixed"
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
INPUT_FASTA_DIR = REPO_ROOT / "Sequences/SC2_by_year_for_PLANT/Transfer-6tvEN29f8yk9AJZ4"
OUTPUT_DIR = REPO_ROOT/ "Sequences/SC2_by_year_for_PLANT/results"

REFERENCE_SEQ = (
    "QKIPGNDNSTATLCLGHHAVPNGTIVKTITNDRIEVTNATELVQNSSIGKICNSPHQILDGGNCTLIDALLGDPQCDGFQNKEWDLFVERSRANSSCYPYDVPDYASLRSLVASSGTLEFKDESFNWTGVKQNGKSSACKRGSSSSFFSRLNWLTSLNNIYPAQNVTMPNKEQFDKLYIWGVHHPDTDKNQFSLFAQSSGRITVSTTRSQQAVIPNIGSRPRVRDIPSRISIYWTIVKPGDILLINSTGNLIAPRGYFKIRSGKSSIMRSDAPIGECKSECITPNGSIPNDKPFQNVNRITYGACPRYVKQSTLKLATGMRNVPEKQTR"
)
TARGET_LENGTH = len(REFERENCE_SEQ)
SCALE_FACTOR = 8
DEFAULT_BATCH_SIZE = 64

ALIGN_IDENTITY_FAIL = 0.80
ALIGN_REF_COVERAGE_FAIL = 0.95
INVALID_AA_REGEX = "X|B|\\*"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PLANT embeddings for every FASTA in a directory and write one CSV per input FASTA."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_FASTA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--suffixes", nargs="+", default=["*.fa", "*.fasta", "*.faa"])
    return parser.parse_args()


def find_file(patterns: list[str]) -> str | None:
    for pattern in patterns:
        matches = glob.glob(str(LOCAL_HF_DIR / "**" / pattern), recursive=True)
        if matches:
            return matches[0]
    return None


def load_plant_runtime() -> tuple[AutoTokenizer, semanticESM, torch.device, bool]:
    ckpt_dir = LOCAL_HF_DIR / SUBFOLDER
    single_path = ckpt_dir / "model.safetensors"
    shard_paths = sorted(glob.glob(str(ckpt_dir / "model-*-of-*.safetensors")))
    assert single_path.exists() or shard_paths, f"safetensors not found under {ckpt_dir}"

    virus_enc_path = find_file(["virus_encoder.joblib"])
    ref_enc_path = find_file(["ref_encoder.joblib"])
    vp_enc_path = find_file(["vp_encoder.joblib"])
    rp_enc_path = find_file(["rp_encoder.joblib"])

    for path_value, encoder_name in [
        (virus_enc_path, "virus"),
        (ref_enc_path, "ref"),
        (vp_enc_path, "vp"),
        (rp_enc_path, "rp"),
    ]:
        assert path_value is not None, f"{encoder_name}_encoder.joblib not found under {LOCAL_HF_DIR}"

    set_encoders(
        joblib.load(virus_enc_path),
        joblib.load(ref_enc_path),
        joblib.load(vp_enc_path),
        joblib.load(rp_enc_path),
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    esm_config = EsmConfig.from_pretrained(MODEL_NAME, use_safetensors=True)
    model = semanticESM.from_pretrained(
        str(ckpt_dir),
        config=esm_config,
        esm_model_name=MODEL_NAME,
        intermediate_dim=256,
        intermediate_dim_encoder=64,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    model.to(device)
    model.eval()
    if use_fp16:
        model.half()

    return tokenizer, model, device, use_fp16


def fasta_to_dataframe(fasta_path: Path) -> pd.DataFrame:
    rows = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        rows.append({"sequence_id": record.id, "seq": str(record.seq)})
    return pd.DataFrame(rows)


def trim_and_filter_sequences(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    working = df.copy()
    working["seq_raw"] = working["seq"]
    working["seq"] = working["seq"].apply(
        lambda seq: plant_trim_to_target_length(seq, REFERENCE_SEQ, TARGET_LENGTH, return_start_pos=False)
    )
    working = working[working["seq"].notna()].copy()

    if working.empty:
        return working

    working = working[~working["seq"].str.contains(INVALID_AA_REGEX, regex=True).fillna(False)].copy()
    working = working[working["seq"].str.len() == TARGET_LENGTH].copy()

    if working.empty:
        return working

    working["identity_to_reference"] = working["seq"].apply(
        lambda seq: plant_sequence_identity(seq, REFERENCE_SEQ)
    )
    working = working[working["identity_to_reference"] >= ALIGN_IDENTITY_FAIL].copy()

    if working.empty:
        return working

    coverage_df = working["seq_raw"].apply(lambda seq: pd.Series(plant_alignment_coverage_metrics(seq, REFERENCE_SEQ)))
    working = pd.concat([working, coverage_df], axis=1)
    working = working[working["aligned_ref_coverage"] >= ALIGN_REF_COVERAGE_FAIL].copy()

    return working.reset_index(drop=True)


def embed_dataframe(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: semanticESM,
    use_fp16: bool,
    batch_size: int,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["sequence_id", "X", "Y", "Z"])

    encoded = tokenize_sequences(df["seq"].tolist(), tokenizer, TARGET_LENGTH)
    dataset = TextDataset(encoded)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embedding_array = embed_sequences(model, loader, use_fp16=use_fp16)

    embedded = pd.DataFrame(
        {
            "sequence_id": df["sequence_id"].tolist(),
            "X": embedding_array[:, 0] * SCALE_FACTOR,
            "Y": embedding_array[:, 1] * SCALE_FACTOR,
            "Z": embedding_array[:, 2] * SCALE_FACTOR,
        }
    )
    return embedded


def iter_fasta_paths(input_dir: Path, suffixes: list[str]) -> list[Path]:
    fasta_paths: list[Path] = []
    for suffix in suffixes:
        fasta_paths.extend(sorted(input_dir.glob(suffix)))
    unique_paths = sorted({path.resolve() for path in fasta_paths})
    return [Path(path) for path in unique_paths if Path(path).is_file()]


def process_fasta_file(
    fasta_path: Path,
    output_dir: Path,
    tokenizer: AutoTokenizer,
    model: semanticESM,
    use_fp16: bool,
    batch_size: int,
) -> None:
    print(f"Processing {fasta_path.name}...")
    raw_df = fasta_to_dataframe(fasta_path)
    print(f"  Input sequences: {len(raw_df)}")

    filtered_df = trim_and_filter_sequences(raw_df)
    print(f"  Retained after trimming/QC: {len(filtered_df)}")

    embedded_df = embed_dataframe(filtered_df, tokenizer, model, use_fp16, batch_size)
    output_path = output_dir / f"{fasta_path.stem}.csv"
    embedded_df.to_csv(output_path, index=False)
    print(f"  Wrote {output_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fasta_paths = iter_fasta_paths(args.input_dir, args.suffixes)
    if not fasta_paths:
        raise FileNotFoundError(f"No FASTA files found in {args.input_dir}")

    print(f"Loading PLANT runtime from {LOCAL_HF_DIR}...")
    tokenizer, model, device, use_fp16 = load_plant_runtime()
    print(f"Running on device: {device}")
    print(f"Found {len(fasta_paths)} FASTA files in {args.input_dir}")

    for fasta_path in fasta_paths:
        process_fasta_file(
            fasta_path=fasta_path,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            model=model,
            use_fp16=use_fp16,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()