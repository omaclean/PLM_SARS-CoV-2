import json
import re
import copy

def parse_nextclade_tree_nuc(json_path):
    """
    Traverses the Auspice tree JSON to accumulate nucleotide mutations
    from the root to each terminal node.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    tree = data.get('tree', {})
    terminal_nodes = {}

    def traverse(node, current_muts):
        node_muts = copy.deepcopy(current_muts)
        
        # Extract nucleotide mutations (e.g., C241T, T11288-)
        branch_muts = node.get("branch_attrs", {}).get("mutations", {}).get("nuc", [])
        for mut in branch_muts:
            match = re.match(r"([A-Z\-\*])(\d+)([A-Z\-\*])", mut)
            if match:
                ref, pos, alt = match.groups()
                node_muts[int(pos)] = alt

        node_name = node.get("name", "Unknown")

        if "children" not in node or not node["children"]:
            terminal_nodes[node_name] = node_muts
        else:
            for child in node.get("children", []):
                traverse(child, node_muts)

    traverse(tree, {})
    return terminal_nodes

def generate_genome_alignment(lineage_data, ref_seq, output_fasta="genome_alignment.fasta"):
    """
    Applies the nucleotide mutation dictionary to the reference genome
    to build an aligned FASTA file.
    """
    with open(output_fasta, 'w') as f:
        f.write(">Wuhan-Hu-1_Reference\n")
        f.write(f"{ref_seq}\n")
        
        for name, muts in lineage_data.items():
            seq_list = list(ref_seq)
            
            for pos, alt in muts.items():
                if 1 <= pos <= len(seq_list):
                    seq_list[pos - 1] = alt
            
            mutated_seq = "".join(seq_list)
            f.write(f">{name}\n")
            # Fold sequence to 80 characters per line for standard FASTA formatting
            for i in range(0, len(mutated_seq), 80):
                f.write(f"{mutated_seq[i:i+80]}\n")

if __name__ == "__main__":
    input_json = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_tree.json"
    output_prefix = '/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_lineage_references'
    output_fasta = f"{output_prefix}_genome_alignment.fasta"
    
    # You will need to paste the full 29903 bp Wuhan-Hu-1 reference genome here.
    # (NCBI Accession NC_045512.2 or MN908947.3)
    #read from /home3/oml4h/PLM_SARS-CoV-2/Sequences/Wuhan_full.fa
    
    with open("/home3/oml4h/PLM_SARS-CoV-2/Sequences/Wuhan_full.fa", "r") as f:
        wuhan_genome_ref = f.read().strip().split("\n", 1)[1].replace("\n", "")
    
    print("Analysing tree and accumulating nucleotide mutations...")
    extracted_data = parse_nextclade_tree_nuc(input_json)
    
    print(f"Extracted data for {len(extracted_data)} terminal lineages.")
    
    generate_genome_alignment(extracted_data, wuhan_genome_ref, output_fasta)
    print(f"Exported alignment to {output_fasta}")