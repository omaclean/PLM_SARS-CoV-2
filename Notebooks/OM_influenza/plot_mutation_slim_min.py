import sys
sys.path.append('../../')

from Functions_HuggingFace import *

import re
import os
from Bio import PDB, Align
from Bio.SeqUtils import seq1


pdb_path = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/4WE4_assembly.pdb"
pdb_path= "/home3/oml4h/PLM_SARS-CoV-2/Sequences/4O5N-assembly1.cif"
pdb_path="/home3/oml4h/PLM_SARS-CoV-2/Sequences/4WE8-assembly1.cif"
pdb_path="/home3/oml4h/PLM_SARS-CoV-2/Sequences/6WXB-assembly1.cif"

sequences = read_sequences_to_dict(
    "/home3/oml4h/PLM_SARS-CoV-2/Sequences/huH3N2_HA_CDS.translated_OM_synth_extra_steps.fas"
)


def _extract_pdb_chain_sequences(pdb_file):
    if pdb_file.lower().endswith((".cif", ".mmcif")):
        parser = PDB.MMCIFParser(QUIET=True)
    else:
        parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_file)
    chain_data = {}
    for model in structure:
        for chain in model:
            seq_chars = []
            residue_ids = []
            for residue in chain:
                if PDB.is_aa(residue):
                    try:
                        aa = seq1(residue.get_resname())
                    except Exception:
                        aa = "X"
                    seq_chars.append(aa)
                    residue_ids.append(residue.get_id())
            if seq_chars:
                chain_data[chain.id] = ("".join(seq_chars), residue_ids)
    return chain_data


def _alignment_indices(alignment):
    try:
        return alignment.indices[0], alignment.indices[1]
    except AttributeError:
        user_indices = []
        pdb_indices = []
        for (u_start, u_end), (p_start, p_end) in zip(*alignment.aligned):
            user_indices.extend(range(u_start, u_end))
            pdb_indices.extend(range(p_start, p_end))
        return user_indices, pdb_indices


def summarize_pdb_alignment(pdb_file, user_sequence, mutation_list=None, threshold_score=50):
    chain_data = _extract_pdb_chain_sequences(pdb_file)
    mutation_list = mutation_list or []
    mutation_positions = []
    for mut in mutation_list:
        match = re.search(r"(\d+)", mut)
        if match:
            mutation_positions.append((mut, int(match.group(1))))

    alignment_maps = {}

    for chain_id, (pdb_seq, residue_ids) in chain_data.items():
        alignment = align_sequences(user_sequence, pdb_seq, mode="local", open_gap_score=-10, extend_gap_score=-0.5)
        if alignment.score < threshold_score:
            continue

        user_indices, pdb_indices = _alignment_indices(alignment)
        if not user_indices:
            continue

        user_min = min(user_indices) + 1
        user_max = max(user_indices) + 1
        pdb_min = min(pdb_indices)
        pdb_max = max(pdb_indices)
        pdb_min_res = residue_ids[pdb_min][1]
        pdb_max_res = residue_ids[pdb_max][1]

        print(f"Aligned region (chain {chain_id}): user {user_min}-{user_max} -> PDB {pdb_min_res}-{pdb_max_res}")

        user_to_pdb = dict(zip(user_indices, pdb_indices))
        alignment_maps[chain_id] = (user_to_pdb, residue_ids)

    if mutation_positions:
        print("\nMutation mapping to PDB (by chain):")
        for mut, pos in mutation_positions:
            mapped = False
            for chain_id, (user_to_pdb, residue_ids) in alignment_maps.items():
                user_idx = pos - 1
                if user_idx in user_to_pdb:
                    pdb_idx = user_to_pdb[user_idx]
                    pdb_resnum = residue_ids[pdb_idx][1]
                    print(f"  {mut}: chain {chain_id} -> PDB residue {pdb_resnum}")
                    mapped = True
            if not mapped:
                print(f"  {mut}: OUTSIDE aligned PDB region")

    return alignment_maps


def flag_outside_mutations(mutation_list, alignment_maps):
    mapped_positions = set()
    for user_to_pdb, _ in alignment_maps.values():
        mapped_positions.update(user_to_pdb.keys())

    flagged = []
    for mut in mutation_list:
        match = re.search(r"(\d+)", mut)
        if not match:
            flagged.append(mut)
            continue
        pos = int(match.group(1)) - 1
        if pos in mapped_positions:
            flagged.append(mut)
        else:
            flagged.append(f"{mut} (OUTSIDE)")
    return flagged


if __name__ == "__main__":
    ids = list(sequences.keys())
    reference_id = ids[3]
    target_id = ids[-1]
    print(f"Reference ID: {reference_id}")
    print(f"Target ID: {target_id}")
    user_seq = sequences[target_id]
    mutations = [
        m
        for m in get_mutations(sequences[reference_id], sequences[target_id])
        if "del" not in m and "-" not in m
    ]

    alignment_maps = summarize_pdb_alignment(pdb_path, user_seq, mutation_list=mutations)
    mutations_flagged = flag_outside_mutations(mutations, alignment_maps)

    view = visualise_mutations_on_pdb(
        pdb_path,
        user_seq,
        mutations_flagged,
        title=f"{target_id} mutations",
    )
    view.show()

    output_dir = "/home3/oml4h/PLM_SARS-CoV-2/Results/structure_play"
    os.makedirs(output_dir, exist_ok=True)
    pdb_name = os.path.basename(pdb_path)
    lineage = target_id.split("|")[-1]
    output_path = os.path.join(
        output_dir,
        f"{pdb_name}_{lineage}_mutations_structure.html",
    )
    with open(output_path, "w") as f:
        f.write(view._make_html())
    print(f"Saved PDB plot to: {output_path}")
    
    
