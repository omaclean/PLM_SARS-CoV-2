from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import numpy as np
import torch
import esm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from Bio import SeqIO
from scipy.special import softmax
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import py3Dmol
import re
from Bio import PDB, Align
from Bio.SeqUtils import seq1
from IPython.display import display, HTML
from Bio.PDB import PDBIO
from io import StringIO
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple, Any, Union, Iterable

import glob
import os
import re
from Bio import pairwise2


## Compression Functions ########################################################
import bz2
import pickle
import _pickle as cPickle

def compressed_pickle(title, data):
    """Compress and serialize arbitrary Python data to disk.

    Args:
            title (str): File path (without extension) where the pickle will be saved.
            data (Any): Python object to serialize.

    Returns:
            None
    """
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)

def decompress_pickle(file):
    """Load a bz2-compressed pickle created by ``compressed_pickle``.

    Args:
            file (str): Path to the ``.pbz2`` file.

    Returns:
            Any: The deserialized Python object.
    """
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data
##################################################################################
## Genbank Annotation Functions ##################################################
def makeOrfTable(genbank_record):
    """Build a dataframe describing each CDS ORF in a GenBank record.

    Args:
        genbank_record (SeqRecord): Biopython GenBank record containing CDS features.

    Returns:
        pandas.DataFrame: Indexed by ORF name with start, end, part, and location columns.
    """
    orfs=[]
    for feature in genbank_record.features:
        if feature.type =="CDS":
            orf = feature.qualifiers['gene'][0]
            for i, locations in enumerate(feature.location.parts):
                orfs.append([orf, locations.start, locations.end, i, locations])
    orfs = pd.DataFrame(orfs)
    orfs.columns = ['ORF','Start','End','Part','Locations']
    orfs = orfs.set_index("ORF")
    return orfs

def makeMatProteinTable(genbank_record):
    """Create a dataframe of mature peptide annotations from a GenBank record.

    Args:
        genbank_record (SeqRecord): Biopython GenBank record containing ``mat_peptide`` features.

    Returns:
        pandas.DataFrame: Indexed by protein name with ORF, coordinates, and location parts.
    """
    proteins=[]
    for feature in genbank_record.features:
        if feature.type =="mat_peptide":
            protein = feature.qualifiers['product'][0]
            orf = feature.qualifiers['gene'][0]
            for i, locations in enumerate(feature.location.parts):
                proteins.append([protein, orf ,locations.start, locations.end, i, locations])
    proteins = pd.DataFrame(proteins)
    proteins.columns = ['Protein',"ORF",'Start','End','Part','Locations']
    proteins = proteins.set_index("Protein")
    return proteins

###################################################################################
## Mutation Functions #############################################################
def mutate_sequence(reference_sequence,mutations):
    """Apply point mutations to a reference protein sequence.

    Args:
        reference_sequence (str): Original amino-acid sequence (1-indexed positions).
        mutations (Iterable[str]): Strings like ``A12T`` describing source AA, position, and target AA.

    Returns:
        str: The mutated amino-acid sequence.
    """
    mutated_seq = reference_sequence
    for mutation in mutations:
        if 'ins' not in mutation and 'del' not in mutation and "X" not in mutation:
            mutant_amino = mutation[-1]
            mutant_pos = int(mutation[1:-1])
            mutated_seq = mutated_seq[:mutant_pos-1]+mutant_amino+mutated_seq[mutant_pos:]
    return mutated_seq

def DMS(reference):
    """Generate all single–amino acid mutants for a reference sequence.

    Args:
        reference (str): Amino-acid sequence used as the background.

    Returns:
        list[SeqRecord]: Biopython records for every single substitution mutant.
    """
    seq_list = []
    amino_acids = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    for i,ref_amino_acid in enumerate(reference):
        for mutant_amino_acid in amino_acids:
            mutated_seq = reference[:i]+mutant_amino_acid+reference[i+1:]
            seq = SeqRecord(Seq(mutated_seq), id=ref_amino_acid+str(i+1)+mutant_amino_acid)
            seq_list.append(seq)
    return seq_list

def DMS_Table(reference):
    """Return a dataframe mapping mutation labels to mutant sequences.

    Args:
        reference (str): Amino-acid reference sequence.

    Returns:
        pandas.DataFrame: Columns ``Mutations`` and ``Sequence`` for each single mutant.
    """
    seq_list = []
    amino_acids = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    for i,ref_amino_acid in enumerate(reference):
        for mutant_amino_acid in amino_acids:
            mutated_seq = reference[:i]+mutant_amino_acid+reference[i+1:]
            seq = SeqRecord(Seq(mutated_seq), id=ref_amino_acid+str(i+1)+mutant_amino_acid)
            seq_list.append([ref_amino_acid+str(i+1)+mutant_amino_acid,seq])
    return pd.DataFrame(seq_list, columns=['Mutations','Sequence'])

#####################################################################################
## Translation Functions ############################################################
def iterative_translate(sequence,truncate_proteins=False):
    """Translate a nucleotide sequence codon-by-codon while handling gaps.

    Args:
        sequence (str | Seq): Nucleotide sequence (can include gaps or ambiguous bases).
        truncate_proteins (bool): If True, truncate translation at the first stop codon.

    Returns:
        str: Amino-acid sequence ("-" for gap codons, "X" for partial codons).
    """
    amino_acid = ""
    for i in range(0,len(sequence)-2,3):
        codon = str(sequence[i:i+3])
        codon = codon.replace("?", "N")
        if "-" in codon:
            if codon == "---":
                amino_acid +="-"
            else:
                amino_acid+= "X"
        else:
            amino_acid += str(Seq(codon).translate())
    if truncate_proteins == True:
        if "*" in amino_acid:
            amino_acid = amino_acid[:amino_acid.index("*")]
    return amino_acid

def translate_with_genbank(sequence,ref):
    """Translate every CDS from a GenBank annotation using an input nucleotide sequence.

    Args:
        sequence (Seq): Nucleotide sequence aligned to the GenBank reference.
        ref (SeqRecord): GenBank record providing CDS coordinates.

    Returns:
        dict: Mapping ``"ORF:Part"`` to dictionaries containing translated sequences and ORF names.
    """
    orfs = makeOrfTable(ref)
    translated_sequence = {orfs.index[i]+":"+str(orfs.iloc[i].Part):{"Sequence":"".join(list(iterative_translate("".join(orfs.iloc[i].Locations.extract(sequence)),truncate_proteins=True))),"ORF":orfs.index[i]} for i in range(len(orfs))}
    return translated_sequence

def translate_mat_proteins_with_genbank(sequence,ref):
    """Translate mature peptide annotations from a GenBank record.

    Args:
        sequence (Seq): Nucleotide sequence aligned to the GenBank reference.
        ref (SeqRecord): GenBank record with ``mat_peptide`` annotations.

    Returns:
        dict: Mapping peptide names to translated sequence strings and metadata.
    """
    proteins = makeMatProteinTable(ref)
    proteins = proteins.drop_duplicates(subset=["ORF",'Start','End','Part',],keep="first")
    proteins_dict={}
    for i in range(len(proteins)):
        protein = "".join(list(iterative_translate("".join(proteins.iloc[i].Locations.extract(sequence)),truncate_proteins=True)))
        if proteins.index[i] in proteins_dict:
            proteins_dict[proteins.index[i]]["Sequence"] = proteins_dict[proteins.index[i]]["Sequence"]+protein
        else:
            proteins_dict[proteins.index[i]] = {"Sequence":protein, "ORF":proteins.iloc[i].ORF, "Part":proteins.iloc[i].Part}
    # translated_sequence = {proteins.index[i]:{"Sequence":"".join(list(iterative_translate("".join(proteins.iloc[i].Locations.extract(sequence)),truncate_proteins=True))), "ORF":proteins.iloc[i].ORF} }
    return proteins_dict
#####################################################################################
## ESM Embedding Functions ##########################################################
def embed_sequence(sequence, model, device, model_layers, batch_converter, alphabet):
    """Embed a protein sequence with either FAIR-ESM or HuggingFace-style ESM APIs.

    Args:
        sequence (str): Amino-acid sequence with no padding tokens.
        model (PreTrainedModel): HuggingFace ESM model instance.
        device (torch.device): Device where tensors should reside.
        model_layers (int): Hidden-state index to extract (falls back to last layer if too large).
        batch_converter (callable): Function that converts (label, seq) pairs to model tokens.
        alphabet (Any): Alphabet object exposing ``padding_idx`` for token trimming.

    Returns:
        tuple: ``(results, base_logits, base_mean_embedding, full_embedding)`` where logits are
        log-softmax probabilities and embeddings are torch tensors on CPU.
    """
    # Sequences to embed
    sequence_data = [('base', sequence)]
    
    # Get tokens
    batch_labels, batch_strs, batch_tokens = batch_converter(sequence_data)
    batch_len = (batch_tokens != alphabet.padding_idx).sum(1)[0]

    # Move tokens to GPU
    if torch.cuda.is_available():
        batch_tokens = batch_tokens.to(device=device, non_blocking=True)

    with torch.no_grad():
        # Choose compatible forward kwargs based on model.forward signature.
        # This avoids passing unsupported args such as output_hidden_states to FAIR-style models.
        forward_params = set()
        try:
            forward_fn = getattr(model, "forward", None)
            if forward_fn is not None:
                forward_params = set(forward_fn.__code__.co_varnames)
        except Exception:
            forward_params = set()

        results = None

        # FAIR-ESM style API
        if "repr_layers" in forward_params:
            fair_kwargs = {"repr_layers": [model_layers]}
            if "return_contacts" in forward_params:
                fair_kwargs["return_contacts"] = False
            results = model(batch_tokens, **fair_kwargs)

            token_representation = results["representations"][model_layers][0]
            logits_tensor = results["logits"][0]

        else:
            # HuggingFace-style API (or wrappers): pass only supported kwargs.
            hf_kwargs = {}
            if "output_hidden_states" in forward_params:
                hf_kwargs["output_hidden_states"] = True

            results = model(batch_tokens, **hf_kwargs)

            hidden_states = getattr(results, "hidden_states", None)
            if hidden_states is None and isinstance(results, dict):
                hidden_states = results.get("hidden_states", None)

            if hidden_states is not None and len(hidden_states) > 0:
                if model_layers < len(hidden_states):
                    token_representation = hidden_states[model_layers][0]
                else:
                    print(
                        f"Warning: Requested layer {model_layers} is out of bounds for this model. "
                        "Using last layer."
                    )
                    token_representation = hidden_states[-1][0]
            else:
                # Fallback for outputs exposing last_hidden_state only.
                if hasattr(results, "last_hidden_state"):
                    token_representation = results.last_hidden_state[0]
                elif isinstance(results, dict) and "last_hidden_state" in results:
                    token_representation = results["last_hidden_state"][0]
                else:
                    raise ValueError(
                        "Model output does not include representations/hidden_states/last_hidden_state."
                    )

            if hasattr(results, "logits"):
                logits_tensor = results.logits[0]
            elif isinstance(results, dict) and "logits" in results:
                logits_tensor = results["logits"][0]
            else:
                raise ValueError("Model output does not include logits.")
    del batch_tokens

    full_embedding = token_representation[1:batch_len - 1].cpu()
    base_mean_embedding = token_representation[1 : batch_len - 1].mean(0).cpu()

    lsoftmax = torch.nn.LogSoftmax(dim=1)
    base_logits = lsoftmax(logits_tensor.to(device="cpu"))
    
    return results, base_logits, base_mean_embedding, full_embedding

def process_protein_sequence(sequence, model, model_layers, batch_converter, alphabet, device):
    """Embed a protein sequence and package logits, mean embeddings, and scores.

    Args:
        sequence (str): Amino-acid sequence to embed.
        model (PreTrainedModel): HuggingFace ESM model.
        model_layers (int): Hidden-state index to harvest.
        batch_converter (callable): Tokenizer helper returned by ``alphabet.get_batch_converter``.
        alphabet (Any): Alphabet object used for token lookup.
        device (torch.device): Device for intermediate tensors.

    Returns:
        dict: ``{"Mean_Embedding": list, "Logits": list, "sequence_grammaticality": float}``.
    """
    # Embed Sequence
    # This will now call the updated function defined in this cell, not the one in Functions.py
    results, base_logits, base_mean_embedding, full_embedding = embed_sequence(
        sequence, model, device, model_layers, batch_converter, alphabet
    )
    
    results_dict = {}
    results_dict["Mean_Embedding"] = base_mean_embedding.tolist()
    results_dict["Logits"] = base_logits.tolist()
    
    # Recalculate grammaticality if needed, or pass through
    # Assuming get_sequence_grammaticality is available in your scope
    try:
        results_dict["sequence_grammaticality"] = get_sequence_grammaticality(sequence, base_logits, alphabet)
    except NameError:
        print("Warning: get_sequence_grammaticality function not found in current scope.")
        results_dict["sequence_grammaticality"] = None
        
    return results_dict


def embed_protein_sequences(protein_sequences,reference_protein,coding_region_name,model,model_layers,device,batch_converter,alphabet,scores=False):
    """Embed reference and mutant protein sequences and optionally compute mutation scores.

    Args:
        protein_sequences (Iterable[Tuple[str,str]]): (label, sequence) pairs to evaluate.
        reference_protein (str): Reference sequence used for scoring and logits extraction.
        coding_region_name (str): Label for the coding region (e.g., ``"S:0"``).
        model (PreTrainedModel): HuggingFace ESM model.
        model_layers (int): Hidden-state index used for embeddings.
        device (torch.device): Device for computation.
        batch_converter (callable): Token converter from the alphabet.
        alphabet (Any): Alphabet helper exposing tokens and indices.
        scores (bool): When True, compute grammaticality/semantic metrics for each mutant.

    Returns:
        dict: Nested dictionary keyed by sequence label then coding region with embeddings/scores.
    """
    #Embed Reference Protein Sequence
    results, reference_logits, reference_mean_embedding, full_embedding = embed_sequence(reference_protein,model,device,model_layers,batch_converter,alphabet)
    
    embeddings = {}
    
    word_pos_prob = {}
    for pos in range(len(reference_protein)):
        for word in alphabet.all_toks:
            word_idx = alphabet.get_idx(word)
            prob = reference_logits[pos + 1, word_idx]
            word_pos_prob[(word, pos)] = prob

    embeddings['Reference'] = {}
    embeddings['Reference'][coding_region_name] = {"Mean_Embedding":reference_mean_embedding.tolist(),
                                        "Logits":reference_logits.tolist(),
                                        "sequence_grammaticality":get_sequence_grammaticality(reference_protein,reference_logits,alphabet),
                                        "sequence_entropy":get_sequence_entropy(reference_protein,reference_logits,alphabet),
                                        "sequence_probabilities":get_reference_probabilities(reference_protein,reference_logits,alphabet)
                                     
                                    }
    #Process Fasta Files       
    fasta_sequences = protein_sequences
    
    i = 0
    print(fasta_sequences)
    for fasta in fasta_sequences:
        print(fasta_sequences)
        name, sequence = fasta[0], str(fasta[1])
        print(name)
        

        #Scores work by creating sequences with no insertions or deletions, then calculating changes from this sequence
        #These embeddings are thus not "real" sequences 
        if scores == True:
            print(len(reference_protein),len(sequence))
            mutations = get_mutations(reference_protein,sequence)
            #get mutation position
            mut_pos= [int(mut[1:-1])-1 for mut in mutations if "del" not in mut ]
            mutation_only_sequence = mutate_sequence(reference_protein,mutations)
            embeddings[name] = {coding_region_name:process_protein_sequence(mutation_only_sequence,model,model_layers,batch_converter,alphabet,device)}
            
            # L1/Manhattan Distance between mean embeddings used for the semantic change
            semantic_change = float(sum(abs(target-base) for target, base in zip(reference_mean_embedding,embeddings[name][coding_region_name]["Mean_Embedding"])))
            gm, ev = grammaticality_and_evolutionary_index(word_pos_prob, reference_protein, mutations)
            print('Semantic score: ', semantic_change)
            print('Grammaticality: ', gm)
            print('Relative Grammaticality: ', ev)
            print('Probability: ', np.exp(gm))
            just_mutations = [mut for mut in mutations if "del" not in mut ]
            mutations_string = str([mut for mut in mutations if "del" not in mut ]).strip("[").strip("]").replace("'", "")
            deletions_string = str([mut for mut in mutations if "del" in mut ]).strip("[").strip("]").replace("'", "")
            
            embeddings[name][coding_region_name]["label"] = name
            embeddings[name][coding_region_name]["semantic_score"] = semantic_change
            embeddings[name][coding_region_name]["grammaticality"] = gm
            embeddings[name][coding_region_name]["relative_grammaticality"] = ev

            #Probability of whole sequence including mutation
            embeddings[name][coding_region_name]['sequence_grammaticality'] = get_sequence_grammaticality(sequence,embeddings[name][coding_region_name]['Logits'],alphabet)
            #narrow sense gramaticallity excludes the mutation- so looks at how whole sequence shifts excluding the focal site
            embeddings[name][coding_region_name]['narrow_sequence_grammaticality'] = get_sequence_grammaticality(sequence,embeddings[name][coding_region_name]['Logits'],alphabet,mask_pos=mut_pos)
            print('Sequence Grammaticality: ', embeddings[name][coding_region_name]['sequence_grammaticality'])
            #Probability ratio between the mutant sequence and the reference sequence
            embeddings[name][coding_region_name]['relative_sequence_grammaticality'] = embeddings[name][coding_region_name]['sequence_grammaticality']-embeddings['Reference'][coding_region_name]['sequence_grammaticality']
            #get probability of all the un-mutated sites
            ref_narrow=get_sequence_grammaticality(reference_protein,reference_logits,alphabet,mask_pos=mut_pos)
            embeddings[name][coding_region_name]['relative_narrow_sequence_grammaticality'] = embeddings[name][coding_region_name]['narrow_sequence_grammaticality']-ref_narrow

            embeddings[name][coding_region_name]["probability"] = np.exp(gm)

            embeddings[name][coding_region_name]["mutation_count"] = len(just_mutations)
            embeddings[name][coding_region_name]["mutations"] = mutations_string
            embeddings[name][coding_region_name]["deletions(not_included_in_scores)"] = deletions_string

            #Entropy of the sequence
            embeddings[name][coding_region_name]["entropy"] = get_sequence_entropy(sequence, embeddings[name][coding_region_name]['Logits'], alphabet)
            embeddings[name][coding_region_name]["sequence_probabilities"] = get_reference_probabilities(sequence, embeddings[name][coding_region_name]['Logits'], alphabet)   
        #If no need for scores, sequences can be embedded as is with insertions, deletions truncations etc
        else:
            embeddings[coding_region_name][name] = process_protein_sequence(str(fasta.seq),model,model_layers,batch_converter,alphabet,device)
        i+=1
    return embeddings

######################################################################################
## Entropy Functions ##################################################################
# Entropy of a sequence is calculated as the sum of the log-probabilities of each amino acid in the sequence
# returns a list of entropies for each position in the sequence
def get_sequence_entropy(sequence, logits, alphabet):
    """
    Calculate the Shannon entropy of the probability distribution at each position in the sequence.
    H(x) = - sum(p(x) * log(p(x)))
    
    Args:
        sequence (str): The amino acid sequence.
        logits (torch.Tensor): The logits from the model (seq_len_with_special_tokens, vocab_size).
        alphabet (esm.data.Alphabet): The alphabet object to map characters to indices.
        
    Returns:
        list: A list of entropy values for each position in the sequence.
    """
    # Ensure logits is a torch.Tensor
    logits_tensor = torch.FloatTensor(logits)
    
    # The logits tensor typically includes <cls> and <eos> tokens at the beginning and end.
    # The actual amino acid sequence corresponds to logits[1 : len(sequence) + 1].
    # We need to slice the logits to only include the actual sequence positions.
    aa_logits = logits_tensor[1 : len(sequence) + 1]
    
    # Convert logits to probabilities
    probs = torch.softmax(aa_logits, dim=-1)
    
    # Calculate entropy: -sum(p * log(p))
    # Use torch.log for the log part and clamp_min for numerical stability to avoid log(0)
    # A common way to calculate this is using torch.special.entr(probs).sum(dim=-1)
    # Or manually:
    log_probs = torch.log(probs.clamp_min(1e-9)) # Clamp to avoid log(0)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    return entropy.tolist()

######################################################################################
## reference probablity for each amino acid
def get_reference_probabilities(sequence, logits, alphabet):
    """
    Calculate the probability of the reference (wild-type) amino acid at each position.
    
    Args:
        sequence (str): The amino acid sequence.
        logits (torch.Tensor): The logits from the model (seq_len_with_special_tokens, vocab_size).
        alphabet (esm.data.Alphabet): The alphabet object to map characters to indices.
        
    Returns:
        list: A list of probabilities for the reference amino acid at each position.
    """
    # Ensure logits is a torch.Tensor
    logits_tensor = torch.FloatTensor(logits)
    
    # Slice logits to match sequence (remove cls and eos)
    aa_logits = logits_tensor[1 : len(sequence) + 1]
    
    # Convert logits to probabilities
    probs = torch.softmax(aa_logits, dim=-1)
    
    # Get indices for the sequence characters
    # alphabet.get_idx() maps char to index
    seq_indices = [alphabet.get_idx(aa) for aa in sequence]
    seq_indices_tensor = torch.tensor(seq_indices)
    
    # Gather probabilities for the specific amino acids in the sequence
    # probs is (seq_len, vocab_size)
    # we want probs[i, seq_indices[i]] for each i
    
    # efficient gathering:
    ref_probs = probs.gather(1, seq_indices_tensor.unsqueeze(1)).squeeze(1)
    
    return ref_probs.tolist()

######################################################################################
## Mutation probability matrix
def get_mutation_prob_matrix(reference_protein, model, model_layers, device, batch_converter, alphabet):
    """
    Embed a reference protein sequence and return a 20 × Length mutation probability matrix.
    Each column represents a position in the sequence, and each row represents one of the 
    20 standard amino acids. Values represent the model's predicted probability of each 
    amino acid at each position.
    
    Args:
        reference_protein (str): Reference amino acid sequence to analyze.
        model (PreTrainedModel): HuggingFace ESM model.
        model_layers (int): Hidden-state index used for embeddings.
        device (torch.device): Device for computation.
        batch_converter (callable): Token converter from the alphabet.
        alphabet (Any): Alphabet helper exposing tokens and indices.
        
    Returns:
        dict: Contains:
            - 'mutation_matrix': numpy.ndarray of shape (20, len(sequence))
            - 'amino_acids': list of 20 amino acid letters (row labels)
            - 'sequence': the reference protein sequence
            - 'positions': list of 1-indexed positions (column labels)
    """
    # Standard amino acids in order
    amino_acids = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    
    # Embed the reference protein sequence
    results, reference_logits, reference_mean_embedding, full_embedding = embed_sequence(
        reference_protein, model, device, model_layers, batch_converter, alphabet
    )
    
    # Slice logits to match sequence (remove cls and eos tokens)
    aa_logits = reference_logits[1 : len(reference_protein) + 1]
    
    # Convert logits to probabilities
    probs = torch.softmax(aa_logits, dim=-1)
    
    # Get indices for the 20 standard amino acids
    aa_indices = [alphabet.get_idx(aa) for aa in amino_acids]
    
    # Extract probabilities for the 20 amino acids at each position
    # probs shape: (seq_len, vocab_size)
    # We want shape: (20, seq_len)
    mutation_matrix = probs[:, aa_indices].T  # Transpose to get (20, seq_len)
    
    # Create 1-indexed position labels
    positions = list(range(1, len(reference_protein) + 1))
    
    return {
        'mutation_matrix': mutation_matrix.numpy(),
        'amino_acids': amino_acids,
        'sequence': reference_protein,
        'positions': positions
    }
######################################################################################
## Scoring Functions #################################################################
def grammaticality_and_evolutionary_index(word_pos_prob, seq, mutations):
    """Calculate log-probability scores and evolutionary ratios for mutations.

    Args:
        word_pos_prob (dict[tuple[str,int], float]): Mapping of (aa, position) to log probability
            from the reference logits (positions are 0-indexed).
        seq (str): Reference amino-acid sequence.
        mutations (Iterable[str]): Mutation strings such as ``A12T`` (no insertions/deletions).

    Returns:
        tuple[float, float]: Sum of mutant log-probabilities and the evolutionary ratio (difference
        between mutant and original log-probabilities).
    """
    if len(mutations) == 0:
        print('No mutations detected')
        return 0, 0
    mut_probs = []
    ev_ratios = []
    current_support = -1
    print('Mutations: ', mutations)
    for mutation in mutations:
        #Ignore insertions
        if 'ins' not in mutation and 'del' not in mutation and "X" not in mutation:
            #Split mutation 
            aa_orig = mutation[0]
            aa_pos = int(mutation[1:-1]) - 1
            aa_mut = mutation[-1]
            if (seq[aa_pos] != aa_orig):
                print(mutation)
            assert(seq[aa_pos] == aa_orig)

            #Get probabilities for changes
            prob_change = word_pos_prob[(aa_mut, aa_pos)]
            prob_original = word_pos_prob[(aa_orig, aa_pos)]
            #Log probabilities to allow for subtraction
            ev_ratio = prob_change - prob_original
            ev_ratios.append(ev_ratio)

            #Log probabilities to allow for sum rather than product
            mut_probs.append(word_pos_prob[(aa_mut, aa_pos)])
    return np.sum(mut_probs), np.sum(ev_ratios)

def get_sequence_grammaticality(sequence,sequence_logits,alphabet,mask_pos=None):   
    """Sum log-probabilities for observed amino acids, optionally masking positions.

    Args:
        sequence (str): Amino-acid sequence being evaluated.
        sequence_logits (array-like): Logits or log-probabilities with bos/eos tokens included.
        alphabet (Any): Alphabet object providing ``get_idx`` for amino acids.
        mask_pos (Iterable[int] | None): Zero-based positions to exclude from the sum.

    Returns:
        float: Log-probability sum over all (or unmasked) positions.
    """
    prob_list = []
    sequence_logits = torch.FloatTensor(sequence_logits)
    for pos in range(len(sequence)):
        word_idx = alphabet.get_idx(sequence[pos])
        word = sequence_logits[(pos + 1,word_idx)]
        prob_list.append(word)
    if mask_pos is None:
        # take probability of whole sequence
        base_grammaticality = np.sum(prob_list)
    else:
        # if "mask" positions given, get the probabilities of observed AAs in sequence except for mutation site tested
        base_grammaticality = np.sum([prob_list[i] for i in range(len(prob_list)) if i not in mask_pos])
    return base_grammaticality

def semantic_calc(target,base):
    """Compute L1 distance between two embedding vectors.

    Args:
        target (Iterable[float]): Target embedding values.
        base (Iterable[float]): Reference embedding values.

    Returns:
        float: Sum of absolute differences.
    """
    return float(sum(abs(np.array(target)-np.array(base) )))
#######################################################################################
## Genbank Functions ##################################################################
def process_sequence_genbank(sequence,genbank,model,model_layers,device,batch_converter,alphabet):
    """Embed all translated proteins defined by a GenBank record.

    Args:
        sequence (Seq): Nucleotide sequence aligned to ``genbank``.
        genbank (SeqRecord): Annotation containing CDS/mature peptide information.
        model (PreTrainedModel): HuggingFace ESM model for embedding.
        model_layers (int): Hidden-state index to retrieve.
        device (torch.device): Device to run inference on.
        batch_converter (callable): Tokenization helper for the model.
        alphabet (Any): Alphabet helper for logits indexing.

    Returns:
        dict: Mapping coding-region names to embeddings/logits/metadata.
    """
    #Translate nucleotide to proteins using genbank
    Coding_Regions= translate_with_genbank(sequence,genbank)
    Mature_Proteins= translate_mat_proteins_with_genbank(sequence,genbank)
    polyprotein_orfs =set([Mature_Proteins[prot]["ORF"] for prot in Mature_Proteins.keys()])
    Filtered_Coding_Regions = {**Coding_Regions}
    for orf in Coding_Regions.keys():
        if Coding_Regions[orf]["ORF"] in polyprotein_orfs:
            del Filtered_Coding_Regions[orf]
    Merged_Coding_Regions = {**Filtered_Coding_Regions,**Mature_Proteins}
    #Embed Sequence
    for key,value in Merged_Coding_Regions.items():
        base_seq = Merged_Coding_Regions[key]["Sequence"]
        results,base_logits, base_mean_embedding, full_embedding = embed_sequence(base_seq,model,device,model_layers,batch_converter,alphabet)
        word_pos_prob = {}
        for pos in range(len(base_seq)):
            for word in alphabet.all_toks:
                word_idx = alphabet.get_idx(word)
                prob = base_logits[pos + 1, word_idx]
                word_pos_prob[(word, pos)] = prob
        value["Mean_Embedding"] = base_mean_embedding.tolist()
        # value["Full_Embedding"] = full_embedding.tolist()
        value["Logits"] = base_logits.tolist()

    all_embeddings = [np.array(Merged_Coding_Regions [key]["Mean_Embedding"]) for key in Merged_Coding_Regions.keys()]
#     Merged_Coding_Regions ["Sum_Embedding"] = list(np.sum(all_embeddings,axis=0))
#     Merged_Coding_Regions ["Concatenated_Embedding"] = list(np.concatenate(all_embeddings))
    return Merged_Coding_Regions

def get_region_from_genbank(sequence,genbank,region_name):
    """Return a single translated region from a GenBank annotation.

    Args:
        sequence (Seq): Nucleotide sequence to translate.
        genbank (SeqRecord): Corresponding GenBank record.
        region_name (str): Key identifying the desired region (matching ``Merged_Coding_Regions``).

    Returns:
        dict: ``{region_name: region_dict}`` containing translated sequence details.
    """
    #Translate nucleotide to proteins using genbank
    Coding_Regions= translate_with_genbank(sequence,genbank)
    Mature_Proteins= translate_mat_proteins_with_genbank(sequence,genbank)
    polyprotein_orfs =set([Mature_Proteins[prot]["ORF"] for prot in Mature_Proteins.keys()])
    Filtered_Coding_Regions = {**Coding_Regions}
    for orf in Coding_Regions.keys():
        if Coding_Regions[orf]["ORF"] in polyprotein_orfs:
            del Filtered_Coding_Regions[orf]
    Merged_Coding_Regions = {**Filtered_Coding_Regions,**Mature_Proteins}
    Merged_Coding_Region = {region_name:Merged_Coding_Regions[region_name]}
    return Merged_Coding_Region



def process_and_dms_sequence_genbank(sequence,genbank,model,model_layers,device,batch_converter,alphabet,specify_orf=""):
    """Embed reference coding regions and all single mutants derived via DMS.

    Args:
        sequence (Seq): Input nucleotide sequence.
        genbank (SeqRecord): Annotation describing coding regions.
        model (PreTrainedModel): HuggingFace ESM model.
        model_layers (int): Hidden-state index to extract.
        device (torch.device): Device to run embeddings on.
        batch_converter (callable): Token conversion callable.
        alphabet (Any): Alphabet helper for logits.
        specify_orf (str): Optional ORF name to restrict processing.

    Returns:
        dict: Nested structure containing reference embeddings and DMS mutant metrics per ORF.
    """
    #Translate nucleotide to proteins using genbank
    Coding_Regions= translate_with_genbank(sequence,genbank)
    Mature_Proteins= translate_mat_proteins_with_genbank(sequence,genbank)
    polyprotein_orfs =set([Mature_Proteins[prot]["ORF"] for prot in Mature_Proteins.keys()])
    Filtered_Coding_Regions = {**Coding_Regions}
    for orf in Coding_Regions.keys():
        if Coding_Regions[orf]["ORF"] in polyprotein_orfs:
            del Filtered_Coding_Regions[orf]
    Merged_Coding_Regions = {**Filtered_Coding_Regions,**Mature_Proteins}
    embeddings = {}
    print(Merged_Coding_Regions.keys())
    if specify_orf !="":
        Merged_Coding_Regions = {specify_orf:Merged_Coding_Regions[specify_orf]}
    #Embed Sequence
    print(Merged_Coding_Regions.keys())
    for key,value in Merged_Coding_Regions.items():
        embeddings[key] = {}
        base_seq = Merged_Coding_Regions[key]["Sequence"]
        results,base_logits, base_mean_embedding, full_embedding = embed_sequence(base_seq,model,device,model_layers,batch_converter,alphabet)
        word_pos_prob = {}
        for pos in range(len(base_seq)):
            for word in alphabet.all_toks:
                word_idx = alphabet.get_idx(word)
                prob = base_logits[pos + 1, word_idx]
                word_pos_prob[(word, pos)] = prob
        embeddings[key]["Reference"] = {"Mean_Embedding":base_mean_embedding.tolist(),
                                        "Logits":base_logits.tolist(),
                                        "sequence_grammaticality":get_sequence_grammaticality(base_seq,base_logits,alphabet)
                                     }
        # Now DMS the sequence and embed and measure to reference
        sequences = DMS(base_seq)
        for fasta in sequences:
            name, sequence = fasta.id, str(fasta.seq)
            print(key,name)
            mutations = [name]
            embeddings[key][name] = process_protein_sequence(sequence,model,model_layers,batch_converter,alphabet,device)
            # L1/Manhattan Distance between mean embeddings used for the semantic change
            semantic_change = float(sum(abs(target-base) for target, base in zip(embeddings[key]["Reference"]["Mean_Embedding"],
                                                                                 embeddings[key][name] ["Mean_Embedding"])))
            gm, ev = grammaticality_and_evolutionary_index(word_pos_prob, base_seq, mutations)
#             print('Semantic score: ', semantic_change)
#             print('Grammaticality: ', gm)
#             print('Relative Grammaticality: ', ev)
            embeddings[key][name]["label"] = name
            embeddings[key][name]["semantic_score"] = semantic_change
            #Probability of mutation, given the reference sequence
            embeddings[key][name]["grammaticality"] = gm
            embeddings[key][name]["relative_grammaticality"] = ev
            #Probability of whole sequence
            embeddings[key][name]['sequence_grammaticality'] = get_sequence_grammaticality(sequence,embeddings[key][name]['Logits'],alphabet)
#             print('Sequence Grammaticality: ', embeddings[key][name]['sequence_grammaticality'])
            #Probability ratio between the mutant sequence and the reference sequence
            embeddings[key][name]['relative_sequence_grammaticality'] = embeddings[key][name]['sequence_grammaticality']-embeddings[key]["Reference"]['sequence_grammaticality']
#             print('Relative Sequence Grammaticality: ', embeddings[key][name]['relative_sequence_grammaticality'])
            embeddings[key][name]["probability"] = np.exp(gm)
#             print(embeddings[key][name]['grammaticality'])
    return embeddings

def read_sequences_to_dict(file_path, file_format="fasta"):
    """Parse sequences from a FASTA/GenBank/etc file into a dictionary.

    Args:
        file_path (str): Path to the sequence file.
        file_format (str): Format understood by ``SeqIO.parse`` (default ``"fasta"``).

    Returns:
        dict[str, str]: Mapping record IDs to plain string sequences.
    """
    sequences = {}
    for record in SeqIO.parse(file_path, file_format):
        sequences[record.id] = str(record.seq)  # Use record.id as the key and record.seq as the value
    return sequences

def get_mutations(seq1, seq2):
    """Compute point mutations or deletions between two aligned sequences.

    Args:
        seq1 (str): Reference sequence (with gaps allowed).
        seq2 (str): Mutated sequence aligned to ``seq1`` length.

    Returns:
        list[str]: Mutation descriptors like ``A12T`` or ``A12del``.
    """
    mutations = []
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            if seq1[i] != '-' and seq2[i] == '-':
                mutations.append('{}{}del'.format(seq1[i], i + 1))
            else:
                mutations.append('{}{}{}'.format(seq1[i] , i + 1, seq2[i]))
    return mutations

def get_indel_mutations(aligned_reference,indel_sequence):
    """Extract substitutions, insertions, and deletions from aligned sequences with indels.

    Args:
        aligned_reference (str): Reference sequence including gaps.
        indel_sequence (str): Sequence containing potential insertions/deletions.

    Returns:
        tuple[list[str], list[str], list[str]]: Point mutations, insertion descriptors (``ins``),
        and deletion descriptors (``del``).
    """
    index = 1
    insertion_subtract = 0
    mutations =[] 
    insertions = []
    deletions = []
    for i,a in enumerate(aligned_reference):
        position = index+i - insertion_subtract
        if a == '-' and indel_sequence[i]!= '-' :
            insertions.append('ins'+str(position)+indel_sequence[i])
            insertion_subtract +=1
        elif a == '-' and indel_sequence[i]== '-' :
            insertion_subtract +=1
        elif  a != '-' and indel_sequence[i] == '-':
            deletions.append('del'+str(position))
        elif aligned_reference[i] != indel_sequence[i]:
            mutations.append(aligned_reference[i]+str(position)+indel_sequence[i])
    return mutations,insertions,deletions

#######################################################################################
## Fasta Functions ####################################################################
def process_fasta(filename,protein_name,reference_sequence,model,model_layers,batch_converter,device,alphabet,insertions=False):
    """Embed each sequence in a FASTA file relative to a reference protein.

    Args:
        filename (str): Path to FASTA file with amino-acid sequences.
        protein_name (str): Identifier used as key in the resulting dictionary.
        reference_sequence (str): Reference amino-acid sequence for scoring.
        model (PreTrainedModel): HuggingFace ESM model instance.
        model_layers (int): Hidden-state index used for embeddings.
        batch_converter (callable): Token conversion helper.
        device (torch.device): Device for computation.
        alphabet (Any): Alphabet helper for logits.
        insertions (bool): If True, skip mutation-based scoring (sequence lengths may differ).

    Returns:
        dict: Embedding and scoring information for the reference and each FASTA entry.
    """
    key = protein_name
    base_seq = reference_sequence  
    embeddings = {key:{}}
    results,base_logits, base_mean_embedding, full_embedding = embed_sequence(base_seq,model,device,model_layers,batch_converter,alphabet)
    # Get position probabilities dictionary
    word_pos_prob = {}
    for pos in range(len(base_seq)):
        for word in alphabet.all_toks:
            word_idx = alphabet.get_idx(word)
            prob = base_logits[pos + 1, word_idx]
            word_pos_prob[(word, pos)] = prob
            
    embeddings[key]["Reference"] = {"Mean_Embedding":base_mean_embedding.tolist(),
                                    "Logits":base_logits.tolist(),
                                    "sequence_grammaticality":get_sequence_grammaticality(base_seq,base_logits,alphabet)}
    for fasta in SeqIO.parse(filename, "fasta"):
        name, sequence = fasta.id, str(fasta.seq)
        if insertions == False:
            mutations = get_mutations(reference_sequence,str(fasta.seq))
        if sequence[-1] == '*':
            sequence = sequence[:-1]
        
        #Remove gap characters
        sequence = sequence.replace('-','')
        
        embeddings[key][name] = process_protein_sequence(sequence,model,model_layers,batch_converter,alphabet,device)
        # L1/Manhattan Distance between mean embeddings used for the semantic change
        semantic_change = float(sum(abs(target-base) for target, base in zip(embeddings[key]["Reference"]["Mean_Embedding"],
                                                                             embeddings[key][name] ["Mean_Embedding"])))
        
        if insertions == False:
            gm, ev = grammaticality_and_evolutionary_index(word_pos_prob, base_seq, mutations)
            #Probability of mutation, given the reference sequence
            embeddings[key][name]["grammaticality"] = gm
            embeddings[key][name]["relative_grammaticality"] = ev
            embeddings[key][name]["probability"] = np.exp(gm)
            print('Grammaticality: ', gm)
            print('Relative Grammaticality: ', ev)
        
        embeddings[key][name]["label"] = name
        embeddings[key][name]["semantic_score"] = semantic_change
        print('Semantic score: ', semantic_change)

        #Probability of whole sequence
        embeddings[key][name]['sequence_grammaticality'] = get_sequence_grammaticality(sequence,embeddings[key][name]['Logits'],alphabet)
        print('Sequence Grammaticality: ', embeddings[key][name]['sequence_grammaticality'])
        #Probability ratio between the mutant sequence and the reference sequence
        embeddings[key][name]['relative_sequence_grammaticality'] = embeddings[key][name]['sequence_grammaticality']-embeddings[key]["Reference"]['sequence_grammaticality']
        print('Relative Sequence Grammaticality: ', embeddings[key][name]['relative_sequence_grammaticality'])
        
    return embeddings
#######################################################################################
## VOC Functions ####################################################################
def build_voc_dictionary(lineage_dict):
    """Map Pango lineages to VOC labels using a provided dictionary.

    Args:
        lineage_dict (dict[str, str | list[str]]): Mapping from variant names to lineage(s).

    Returns:
        dict[str, list[str]]: VOC name to list of matching lineage prefixes.
    """
    vocs = {"Alpha":["B.1.1.7"],"Beta":["B.1.351"],"Gamma":["P.1"],"Delta":["B.1.617.2"],"Omicron":["B.1.1.529"]}
    
    for key in lineage_dict.keys():
        val = lineage_dict[key]
        if type(val) == str:
            val = [val]
            
        for lineage in val:
            if "." in lineage:
                start  = lineage.split(".")[0]
            else:
                start  = lineage
            if "B.1.617.2" == lineage or "B.1.617.2." in lineage or start in vocs["Delta"] :
                vocs["Delta"].append(key)
            elif "B.1.1.7" == lineage or  "B.1.1.7." in lineage or start in vocs["Alpha"]:
                vocs["Alpha"].append(key)
            elif "B.1.351" == lineage or "B.1.351." in lineage or start in vocs["Beta"]:
                vocs["Beta"].append(key)
            elif "P.1" == lineage or  lineage == "P.1" or "P.1." in lineage or start in vocs["Gamma"]:
                vocs["Gamma"].append(key)
            elif "B.1.1.529" == lineage or "B.1.1.529." in lineage or start in vocs["Omicron"]:
                vocs["Omicron"].append(key)
    return vocs

def is_voc(lineage,voc_dictionary):
    """Determine which VOC a lineage belongs to, if any.

    Args:
        lineage (str): Pango lineage string (e.g., ``"B.1.1.7.2"``).
        voc_dictionary (dict[str, list[str]]): Output of ``build_voc_dictionary``.

    Returns:
        str: VOC label or ``"Non-VOC"`` if no match.
    """
    for key,values in voc_dictionary.items():
        if "." in lineage:
            start  = lineage.split(".")[0]
        else:
            start  = lineage
        #If base VOC lineage 
        if lineage in values:
            return key
        #If not the base VOC
        else:
            if start in values:
                return key
            elif values[0]+"." in lineage:
                return key
    return "Non-VOC"


#######################################################################################
## Epistasis Functions ##############################################################
def get_reference_mutations(ref,mut):
    """Return mutations required to transform one aligned sequence into another.

    Args:
        ref (str): Reference sequence with possible gaps.
        mut (str): Target sequence aligned to ``ref``.

    Returns:
        list[str]: Mutation strings, ignoring gap positions in the reference.
    """
    mutations = []
    gap_counter = 0
    for i,a in enumerate(ref):
        if a == "-":
            gap_counter+=1
            continue
        elif a != mut[i]:
            # print(a+str(i+1-gap_counter)+mut[i])
            mutations.append(a+str(i+1-gap_counter)+mut[i])
    return mutations   

def revert_sequence(reference_sequence,mutations):
    """Undo mutations by applying the reference amino acid at each specified position.

    Args:
        reference_sequence (str): Sequence currently containing mutations.
        mutations (Iterable[str]): Mutation strings indicating target positions to revert.

    Returns:
        str: Sequence after reverting each supplied mutation.
    """
    mutated_seq = reference_sequence
    for mutation in mutations:
        if 'ins' not in mutation and 'del' not in mutation and "X" not in mutation:
            mutant_amino = mutation[0]
            mutant_pos = int(mutation[1:-1])
            mutated_seq = mutated_seq[:mutant_pos-1]+mutant_amino+mutated_seq[mutant_pos:]
    return mutated_seq

def format_logits(logits,alphabet):
    """Convert raw logits to a tidy pandas DataFrame of amino-acid probabilities.

    Args:
        logits (array-like): Raw logits including BOS/EOS rows.
        alphabet (Any): Alphabet exposing ``all_toks`` for column ordering.

    Returns:
        pandas.DataFrame: Logits trimmed to amino-acid rows with sequential index.
    """
    amino_acids = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    logits = pd.DataFrame(logits)
    logits.columns = alphabet.all_toks
    logits = logits[amino_acids]
    logits = logits[1:-1]
    logits = logits.reset_index(drop=True)
    return logits

def remap_logits(mutated_logits,ref_seq_aligned,mutated_seq_aligned):
    """Realign logits from a gapless sequence back onto an alignment with gaps.

    Args:
        mutated_logits (pandas.DataFrame): Logits dataframe produced by ``format_logits``.
        ref_seq_aligned (str): Reference sequence including gaps.
        mutated_seq_aligned (str): Mutant sequence including gaps (aligned to reference).

    Returns:
        pandas.DataFrame: Logits dataframe indexed by alignment positions, including gap rows.
    """
    #Add sequence used to produce the logits to the dataframe
    mutated_logits['Sequence'] = list(mutated_seq_aligned.replace('-',''))
    
    
    #Make holder blank row for gaps
    blank_row_for_gap =  pd.DataFrame(np.full(len(mutated_logits.columns),np.nan)).T
    blank_row_for_gap.columns = mutated_logits.columns
    blank_row_for_gap['Sequence'] = '-'
    
    #Counter for gaps in reference sequence to realign to
    sequence_index = 0

    #List of rows to keep
    rows = []

    #Loop through correct logits
    for i in range(len(mutated_logits)):

        if ref_seq_aligned[sequence_index] == '-':
            sequence_index+=1
            continue

        if mutated_seq_aligned[sequence_index] == '-':
            print(ref_seq_aligned[sequence_index],sequence_index,i)
            while(mutated_seq_aligned[sequence_index] == '-'):
                gap_seq = blank_row_for_gap.copy()
                gap_seq.index = [sequence_index]
                rows.append(gap_seq)
                sequence_index+=1

        if mutated_seq_aligned[sequence_index] != '-':
            
            row = pd.DataFrame(mutated_logits.iloc[i,:]).T
            row.index = [sequence_index]
            rows.append(row)
            print(ref_seq_aligned[sequence_index],sequence_index,i,list(row.Sequence)[0])
            sequence_index+=1
            
    #Concatenate rows
    rows = pd.concat(rows,axis=0)
    print("".join(rows.Sequence.values))
    print(sequence_index,i)
    print(len(rows))
    print("".join(rows.Sequence.values) == mutated_seq_aligned.replace("EPE",''))
    return rows

def check_valid (v,min,max):
    """Return value if it lies outside [min, max], else None.

    Args:
            v (float | None): Value to test.
            min (float): Lower threshold.
            max (float): Upper threshold.

    Returns:
            float | None: ``v`` when outside the interval, otherwise ``None``.
    """
    if v < min or v >max:
        return v
    return None

def align_sequences(reference_seq, query_seq, mode='local', open_gap_score=-10, extend_gap_score=-0.5):
    """Perform pairwise sequence alignment using Biopython's PairwiseAligner.
    
    Args:
        reference_seq (str): Reference sequence (can be from FASTA, PDB, etc.)
        query_seq (str): Query sequence to align against reference
        mode (str): Alignment mode - 'local' (default) or 'global'
        open_gap_score (float): Gap opening penalty (default: -10)
        extend_gap_score (float): Gap extension penalty (default: -0.5)
        
    Returns:
        Bio.Align.PairwiseAlignment: Best alignment object with .score, .indices, etc.
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = mode
    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score
    
    alignments = aligner.align(reference_seq, query_seq)
    return alignments[0]  # Return best alignment
  
def visualise_mutations_on_pdb(pdb_file, user_sequence, mutation_list, threshold_score=50, 
                               coordinate_map=None, background_values=None, title=None, canonical_map=None,
                               surface_opacity=None, surface_color="#dddddd", mutation_surface_opacity=None,
                               hide_cartoon=False, mutation_surface_probe_radius=None,
                               base_surface_exclude_mutations=False):
    """
    Maps user-defined mutations onto a PDB structure (trimer friendly).
    
    Args:
        pdb_file (str): Path to .pdb file.
        user_sequence (str): The specific K lineage AA sequence string.
        mutation_list (list): List of strings e.g., ['A123T', 'N145K'].
                              Assumes 1-based indexing in the string.
        threshold_score (int): Minimum alignment score to consider a chain a "match".
        coordinate_map (dict, optional): Mapping from user sequence positions (1-based) to 
                                        canonical/reference positions. E.g., {145: 144, 146: 145}.
                                        If provided, positions in mutation_list will be remapped.
        background_values (dict, optional): Dict mapping positions (1-based) to continuous values
                                           (e.g., dn/ds, surface exposure, entropy). Structure will
                                           be colored by these values using a gradient.
        title (str, optional): Title/label for the visualization and legend (e.g., "PLM Entropy").
        canonical_map (dict, optional): Mapping from 0-based sequence indices to canonical numbering
                           (e.g., H3 numbering: {0: 'SP-15', 16: '1', 173: '158A'}).
                           Displays a separate "Canonical Numbering" legend.
        surface_opacity (float, optional): If provided, add a global molecular surface with this opacity.
        surface_color (str, optional): Hex color for the global surface (default: "#dddddd").
        mutation_surface_opacity (float, optional): Opacity for colored mutation surface patches.
        hide_cartoon (bool, optional): If True, hide the ribbon/cartoon representation.
        mutation_surface_probe_radius (float, optional): Probe radius for mutation surface patches.
        base_surface_exclude_mutations (bool, optional): If True, mask mutation residues from the
            base surface so colored patches aren't shrouded.
    """
    
    # 1. Parse Mutation Indices from the list (e.g. 'A123T' -> 122 (0-based))
    # We assume the input list uses standard 1-based biological numbering
    sites_of_interest = {}
    
    # Color palette for mutations - using matplotlib Dark2 or tab20 colormap

    
    # If more than 8 mutations, use tab20 (20 colors), otherwise use Dark2 (8 colors)
    #    Determine palette based on list length
    if len(mutation_list) > 8:
        # tab20 is a Colormap object; we sample it
        colors = [plt.cm.tab20(i) for i in range(20)]

    else:
        # Dark2 is a Colormap; we sample it
        colors = [plt.cm.Dark2(i) for i in range(8)]
    colors_hex = []
    for color in colors:
        if len(color) == 4:
            r, g, b, _ = color
        else:
            r, g, b = color
        colors_hex.append('#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)))
    mutation_colors = {mut: colors_hex[i % len(colors_hex)] for i, mut in enumerate(mutation_list)}

    for mut in mutation_list:
        match = re.search(r'(\d+)', mut)
        if match:
            # Get 1-based position from mutation string
            pos_1based = int(match.group(1))
            
            # Apply coordinate mapping if provided
            if coordinate_map is not None and pos_1based in coordinate_map:
                mapped_pos = coordinate_map[pos_1based]
                # Convert mapped position to 0-based index
                idx = mapped_pos - 1
            else:
                # Convert 1-based str index to 0-based list index
                idx = pos_1based - 1
            
            sites_of_interest[idx] = mut
    
    # 2. Parse PDB/MMCIF Structure
    if pdb_file.lower().endswith((".cif", ".mmcif")):
        parser = PDB.MMCIFParser(QUIET=True)
    else:
        parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_file)
    
    # Check if monomer (1 chain) or multimer
    # We check chains in the first model to decide if it's a monomer PDB.
    try:
        first_model = next(iter(structure))
        first_model_chains = list(first_model.get_chains())
        is_monomer = len(first_model_chains) == 1
    except StopIteration:
        is_monomer = True

    # Dictionary to store hits: { ChainID : [(ResidueID, MutationLabel), ...] }
    residues_to_highlight = {}

    print(f"Processing PDB: {pdb_file}")
    if is_monomer:
        print("  -> Detected Monomer PDB (will use biological assembly)")
    else:
        print("  -> Detected Multimer PDB (will use file content)")
    
    # 3. Iterate over ALL chains (A, B, C...) to handle Trimers
    for model in structure:
        for chain in model:
            
            # Extract sequence from actual atom coordinates
            pdb_residues = [] # List of (ResNum, InsertionCode)
            pdb_seq_str = ""
            
            for residue in chain:
                if PDB.is_aa(residue):
                    pdb_residues.append(residue.id) # .id is tuple (' ', 145, ' ')
                    pdb_seq_str += seq1(residue.resname)
            
            # Skip empty chains
            if not pdb_seq_str: continue

            # Align User Seq vs Chain Seq using shared alignment function
            alignment = align_sequences(user_sequence, pdb_seq_str, mode='local', 
                                       open_gap_score=-10, extend_gap_score=-0.5)
            
            # Simple score check to see if this chain is the protein of interest (HA1)
            # and not the stem (HA2) or a nanobody
            if alignment.score < threshold_score:
                continue
                
            print(f"  -> Match found on Chain {chain.id} (Score: {alignment.score:.1f})")

            # 4. Map User Indices to PDB Residues using the Alignment
            # We iterate through the alignment trace
            # aligned arrays: [0] is query (user), [1] is target (pdb)
            
            # Get the indices of the matches in both sequences
            try:
                user_indices = alignment.indices[0] # Indices in user_sequence
                pdb_indices = alignment.indices[1]  # Indices in pdb_seq_str
            except AttributeError:
                # Fallback for older Biopython versions (e.g. < 1.80)
                user_indices = []
                pdb_indices = []
                # alignment.aligned is a tuple of two lists of tuples: ([(u_start, u_end), ...], [(p_start, p_end), ...])
                for (u_start, u_end), (p_start, p_end) in zip(*alignment.aligned):
                    user_indices.extend(range(u_start, u_end))
                    pdb_indices.extend(range(p_start, p_end))
            
            # Create a lookup: User_Index -> PDB_Array_Index
            # We only care if the user index is in our sites_of_interest
            user_to_pdb_array_map = dict(zip(user_indices, pdb_indices))
            
            for site_idx, mut_label in sites_of_interest.items():
                if site_idx in user_to_pdb_array_map:
                    pdb_array_idx = user_to_pdb_array_map[site_idx]
                    
                    # Retrieve the REAL PDB residue ID (handling numbering/insertion codes)
                    try:
                        real_residue_id = pdb_residues[pdb_array_idx]
                        resi_num = real_residue_id[1]
                        
                        # Store for plotting
                        if chain.id not in residues_to_highlight:
                            residues_to_highlight[chain.id] = []
                        residues_to_highlight[chain.id].append((resi_num, mut_label))
                        
                    except IndexError:
                        pass

    # 5. Visualise with py3Dmol
    view = py3Dmol.view(width=800, height=600)
    
    # Check if we have multiple models (e.g. biological assembly in separate models)
    # If so, we need to add each model individually to show them all at once
    # instead of as an animation.
    num_models = len(list(structure))
    model_format = "mmcif" if pdb_file.lower().endswith((".cif", ".mmcif")) else "pdb"
    if not is_monomer and num_models > 1:
        print(f"  -> Detected {num_models} models. Adding all models to viewer.")
        io = PDBIO()
        for i, model in enumerate(structure):
            # Write model to string
            s = StringIO()
            io.set_structure(model)
            io.save(s)
            view.addModel(s.getvalue(), model_format)
    else:
        # Use doAssembly=True to generate the biological assembly (e.g. trimer) ONLY if it's a monomer file
        # If the file is already a multimer (single model with multiple chains), we assume it represents the assembly we want.
        view.addModel(open(pdb_file).read(), model_format, {'doAssembly': is_monomer})
    
    # Apply background coloring if provided
    if background_values is not None:
        # Set base style for everything to grey cartoon first
        view.setStyle({'cartoon': {'color': '#eeeeee'}})
        
        # Normalize values to [0, 1] for color mapping
        vals = list(background_values.values())
        min_val = min(vals)
        max_val = max(vals)
        val_range = max_val - min_val if max_val != min_val else 1.0
        
        # Apply gradient coloring to each position with a background value
        for pos_1based, value in background_values.items():
            # Normalize value
            norm_val = (value - min_val) / val_range
            
            # Map to color gradient (blue -> white -> red)
            if norm_val < 0.5:
                # Blue to white
                t = norm_val * 2
                r = int(255 * t)
                g = int(255 * t)
                b = 255
            else:
                # White to red
                t = (norm_val - 0.5) * 2
                r = 255
                g = int(255 * (1 - t))
                b = int(255 * (1 - t))
            
            color_hex = f'#{r:02x}{g:02x}{b:02x}'
            
            # Apply coordinate mapping if provided
            if coordinate_map is not None and pos_1based in coordinate_map:
                mapped_pos = coordinate_map[pos_1based]
            else:
                mapped_pos = pos_1based
            
            # Apply coloring to this residue
            selector = {'resi': int(mapped_pos)}
            view.addStyle(selector, {'cartoon': {'color': color_hex}})
    else:
        # Base style: Grey Cartoon (only if no background coloring)
        view.setStyle({'cartoon': {'color': '#eeeeee'}})

    if hide_cartoon:
        view.setStyle({'cartoon': {'opacity': 0.0}})

    # Optional global molecular surface (base layer)
    if surface_opacity is not None:
        base_surface_selection = None
        if base_surface_exclude_mutations and residues_to_highlight:
            exclude_resis = []
            for residues in residues_to_highlight.values():
                for res_num, _ in residues:
                    exclude_resis.append(int(res_num))
            if exclude_resis:
                base_surface_selection = {"not": {"resi": exclude_resis}}
        view.addSurface(
            py3Dmol.SAS,
            {"opacity": surface_opacity, "color": surface_color},
            base_surface_selection,
        )
    
    # Add style for non-protein atoms (e.g. glycans) - make them visible but neutral
    # 'hetflag': True selects heteroatoms (ligands, water, etc.)
    # We exclude water usually, but let's just style hetatoms.
    view.addStyle({'hetflag': True}, {'stick': {'color': '#cccccc', 'radius': 0.15}})
    
    # Loop through our mapped hits and colour them
    count = 0
    matched_chains = set()
    
    for chain_id, residues in residues_to_highlight.items():
        matched_chains.add(chain_id)
        # Remove duplicates based on residue number, but keep mutation label (assume same residue = same mutation)
        unique_residues = {}
        for r, m in residues:
            unique_residues[r] = m
            
        count += len(unique_residues)
        
        for res_num, mut_label in unique_residues.items():
            color = mutation_colors[mut_label]
            
            # If monomer, we don't specify chain to ensure it applies to the assembly if py3Dmol replicates it?
            # Actually, if doAssembly=True, py3Dmol might create multiple chains.
            # If it's a monomer PDB but biological assembly is a trimer, we want to highlight all copies.
            # If we specify chain='A', it might only highlight chain A.
            # If we DON'T specify chain, it highlights that residue number in ALL chains.
            # So if it's a monomer PDB (1 chain parsed) but we want to show multimer, 
            # we should probably NOT specify chain ID in the style, so it hits all generated copies.
            # BUT if the PDB itself is a multimer (e.g. A, B, C parsed), we must specify chain ID 
            # to avoid highlighting wrong residues in other chains (e.g. HA2 vs HA1).
            
            selector = {'resi': int(res_num)}
            if not is_monomer:
                selector['chain'] = chain_id
            
            view.addStyle(
                selector,
                {'cartoon': {'color': color}} 
            )
            view.addStyle(
                selector,
                {'stick': {'color': color, 'radius': 0.4}} 
            )
            if mutation_surface_opacity is not None:
                patch_opacity = mutation_surface_opacity
            else:
                patch_opacity = surface_opacity if surface_opacity is not None else 0.5
            patch_options = {'opacity': patch_opacity, 'color': color}
            if mutation_surface_probe_radius is not None:
                patch_options['probeRadius'] = mutation_surface_probe_radius
            view.addSurface(
                py3Dmol.SAS,
                patch_options,
                selector
            )

    view.zoomTo()
    print(f"Mapped {count} mutation sites across the structure.")
    print(f"Matched chains: {sorted(list(matched_chains))}")
    
    # Display Legend
    legend_html = "<div style='font-family: monospace; margin-top: 10px;'>"
    
    # Add title if provided
    if title:
        legend_html += f"<h3 style='margin: 5px 0;'>{title}</h3>"
    
    # Add background color scale if applicable
    if background_values is not None:
        vals = list(background_values.values())
        min_val = min(vals)
        max_val = max(vals)
        
        legend_html += "<div style='margin: 10px 0;'>"
        legend_html += f"<b>Background Color Scale:</b> {min_val:.3f} "
        legend_html += "<span style='display: inline-block; width: 100px; height: 15px; "
        legend_html += "background: linear-gradient(to right, #0000ff, #ffffff, #ff0000); "
        legend_html += "border: 1px solid #ccc; vertical-align: middle;'></span>"
        legend_html += f" {max_val:.3f}"
        legend_html += "</div>"
    
    # Add mutation legend
    if mutation_list:
        legend_html += "<div style='margin-top: 10px;'><b>Mutation Legend:</b><br>"
        for mut, color in mutation_colors.items():
            # Try to get canonical name if available
            display_name = mut
            if canonical_map is not None:
                # Check if canonical_map is keyed by mutation string (e.g. 'A123T')
                if mut in canonical_map:
                    display_name = canonical_map[mut]
                else:
                    # Check if canonical_map is keyed by position (e.g. 122)
                    match = re.search(r'\d+', mut)
                    if match:
                        seq_pos_1based = int(match.group())
                        seq_pos_0based = seq_pos_1based - 1 # Assuming map is 0-based
                        
                        canon_pos = None
                        if seq_pos_0based in canonical_map:
                            canon_pos = canonical_map[seq_pos_0based]
                        elif seq_pos_1based in canonical_map:
                            canon_pos = canonical_map[seq_pos_1based]
                            
                        if canon_pos is not None:
                            # If the map value looks like a full mutation (e.g. 'K2N'), use it directly
                            if re.search(r'[A-Z]\d+[A-Z]', str(canon_pos)) or 'HA2' in str(canon_pos):
                                display_name = str(canon_pos)
                            else:
                                # Reconstruct: Ref + CanonPos + Alt
                                # We need original ref/alt from 'mut' string
                                parts = re.match(r'([A-Z])(\d+)([A-Z])', mut)
                                if parts:
                                    display_name = f"{parts.group(1)}{canon_pos}{parts.group(3)}"
                                else:
                                    display_name = str(canon_pos)
            
            legend_html += f"<span style='color: {color}; margin-right: 15px;'>&#9632; {display_name}</span>"
        legend_html += "</div>"
    
    # Add canonical numbering legend if provided
    if canonical_map is not None and mutation_list:
        legend_html += "<div style='margin-top: 15px; border-top: 1px solid #ccc; padding-top: 10px;'>"
        legend_html += "<b>Canonical Numbering:</b><br>"
        legend_html += "<table style='font-size: 0.9em; border-collapse: collapse;'>"
        legend_html += "<tr><th style='text-align: left; padding: 2px 10px;'>Canonical</th>"
        legend_html += "<th style='text-align: left; padding: 2px 10px;'>Seq Position</th>"
        legend_html += "<th style='text-align: left; padding: 2px 10px;'>OG Coordinates</th></tr>"
        
        for mut in mutation_list:
            # Extract position from mutation (e.g., 'A145K' -> 145)
            match = re.search(r'\d+', mut)
            if match:
                seq_pos_1based = int(match.group())
                seq_pos_0based = seq_pos_1based - 1
                
                # Get canonical label if available
                canonical_label = 'N/A'
                
                # Try direct mutation lookup first (e.g. 'A145K')
                if mut in canonical_map:
                    canonical_label = canonical_map[mut]
                # Then try 0-based index
                elif seq_pos_0based in canonical_map:
                    canon_pos = canonical_map[seq_pos_0based]
                    # If the map value looks like a full mutation (e.g. 'K2N'), use it directly
                    if re.search(r'[A-Z]\d+[A-Z]', str(canon_pos)) or 'HA2' in str(canon_pos):
                        canonical_label = str(canon_pos)
                    else:
                        # Reconstruct: Ref + CanonPos + Alt
                        parts = re.match(r'([A-Z])(\d+)([A-Z])', mut)
                        if parts:
                            canonical_label = f"{parts.group(1)}{canon_pos}{parts.group(3)}"
                        else:
                            canonical_label = str(canon_pos)
                # Then try 1-based index (just in case)
                elif seq_pos_1based in canonical_map:
                    canonical_label = canonical_map[seq_pos_1based]
                
                color = mutation_colors.get(mut, '#000000')
                
                legend_html += f"<tr><td style='padding: 2px 10px; color: {color};'>{canonical_label}</td>"
                legend_html += f"<td style='padding: 2px 10px;'>{seq_pos_1based}</td>"
                legend_html += f"<td style='padding: 2px 10px; color: black;'>{mut}</td></tr>"
        
        legend_html += "</table></div>"
    
    legend_html += "</div>"
    display(HTML(legend_html))
    
    # Monkey-patch the view object to include the legend in its HTML output
    # This ensures that when view._make_html() is called (e.g. when saving),
    # the legend is included.
    original_make_html = view._make_html
    
    def make_html_with_legend(*args, **kwargs):
        return original_make_html(*args, **kwargs) + legend_html
        
    view._make_html = make_html_with_legend
    
    return view

def export_view_to_png(view, output_path, width=3000, height=3000, view_state=None, zoom_to=True):
    """
    Export a py3Dmol view to a high-resolution PNG file.

    Args:
        view (py3Dmol.view): The view instance to export.
        output_path (str): Destination PNG file path.
        width (int): Render width in pixels.
        height (int): Render height in pixels.
        view_state (list | None): Optional camera state from ``view.getView()``.
        zoom_to (bool): If True, call ``view.zoomTo()`` before export.

    Returns:
        bool: True if a PNG was written, False otherwise.
    """
    if view_state is not None:
        try:
            view.setView(view_state)
        except Exception:
            pass

    if zoom_to:
        try:
            view.zoomTo()
        except Exception:
            pass

    if hasattr(view, "resize"):
        try:
            view.resize(width, height)
        except Exception:
            pass

    try:
        png_data = view.png()
    except Exception:
        print("PNG export failed: view.png() is unavailable in this environment.")
        return False

    if png_data is None:
        print("PNG export failed: no image data returned.")
        return False

    import base64

    if isinstance(png_data, bytes):
        png_bytes = png_data
    elif isinstance(png_data, str):
        if png_data.startswith("data:image/png;base64,"):
            png_data = png_data.split(",", 1)[1]
        png_bytes = base64.b64decode(png_data)
    else:
        print("PNG export failed: unsupported image data format.")
        return False

    with open(output_path, "wb") as f:
        f.write(png_bytes)
    return True

# def _get_plane_from_points(points):
#     if len(points) < 3:
#         raise ValueError("Need at least 3 points to define a plane.")
#     centroid = np.mean(points, axis=0)
#     centered = points - centroid
#     _, _, vh = np.linalg.svd(centered, full_matrices=False)
#     normal = vh[-1]
#     normal = normal / np.linalg.norm(normal)
#     return centroid, normal

# def _rotation_matrix_from_vectors(vec_a, vec_b):
#     a = vec_a / np.linalg.norm(vec_a)
#     b = vec_b / np.linalg.norm(vec_b)
#     v = np.cross(a, b)
#     c = np.dot(a, b)
#     if np.isclose(c, 1.0):
#         return np.eye(3)
#     if np.isclose(c, -1.0):
#         axis = np.array([1.0, 0.0, 0.0])
#         if np.allclose(a, axis):
#             axis = np.array([0.0, 1.0, 0.0])
#         v = np.cross(a, axis)
#         v = v / np.linalg.norm(v)
#         return _rotation_matrix_from_vectors(a, -a + 2 * v * np.dot(v, a))
#     s = np.linalg.norm(v)
#     vx = np.array(
#         [
#             [0, -v[2], v[1]],
#             [v[2], 0, -v[0]],
#             [-v[1], v[0], 0],
#         ]
#     )
#     r = np.eye(3) + vx + vx.dot(vx) * ((1 - c) / (s * s))
#     return r

# def _rotation_matrix_axis_angle(axis, angle_rad):
#     axis = np.array(axis, dtype=float)
#     axis = axis / np.linalg.norm(axis)
#     x, y, z = axis
#     c = np.cos(angle_rad)
#     s = np.sin(angle_rad)
#     c1 = 1 - c
#     return np.array(
#         [
#             [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
#             [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
#             [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
#         ]
#     )

# def _transform_structure(structure, rotation=None, translation=None, center=None):
#     rotation = np.eye(3) if rotation is None else rotation
#     translation = np.zeros(3) if translation is None else np.array(translation, dtype=float)
#     if center is not None:
#         center = np.array(center, dtype=float)
#         translation = translation + center - rotation.dot(center)
#     structure.transform(rotation, translation)

# def embed_membrane_to_protein_plane(
#     protein_pdb_path,
#     membrane_pdb_path,
#     residue_start=526,
#     residue_end=541,
#     output_membrane_pdb_path=None,
#     rotate_membrane=True,
#     inplane_align=True,
#     inplane_rotation_deg=None,
# ):
#     if protein_pdb_path.lower().endswith((".cif", ".mmcif")):
#         protein_parser = PDB.MMCIFParser(QUIET=True)
#     else:
#         protein_parser = PDB.PDBParser(QUIET=True)
#     protein_structure = protein_parser.get_structure("protein", protein_pdb_path)

#     ca_points = []
#     for model in protein_structure:
#         for chain in model:
#             for residue in chain:
#                 if not PDB.is_aa(residue):
#                     continue
#                 resseq = residue.id[1]
#                 if residue_start <= resseq <= residue_end and "CA" in residue:
#                     ca_points.append(residue["CA"].get_coord())

#     if len(ca_points) < 3:
#         raise ValueError("Not enough CA atoms found for the specified residue range.")

#     ca_points = np.array(ca_points)
#     protein_plane_center, protein_plane_normal = _get_plane_from_points(ca_points)
#     protein_axis = None
#     try:
#         centered = ca_points - protein_plane_center
#         _, _, vh = np.linalg.svd(centered, full_matrices=False)
#         protein_axis = vh[0]
#     except Exception:
#         protein_axis = None

#     protein_all_points = []
#     for model in protein_structure:
#         for chain in model:
#             for residue in chain:
#                 if not PDB.is_aa(residue):
#                     continue
#                 if "CA" in residue:
#                     protein_all_points.append(residue["CA"].get_coord())
#     if protein_all_points:
#         protein_all_points = np.array(protein_all_points)
#         protein_center = protein_all_points.mean(axis=0)
#         tail_vector = protein_plane_center - protein_center
#         if np.dot(protein_plane_normal, tail_vector) < 0:
#             protein_plane_normal = -protein_plane_normal

#     membrane_parser = PDB.PDBParser(QUIET=True)
#     membrane_structure = membrane_parser.get_structure("membrane", membrane_pdb_path)

#     membrane_points = []
#     phosphate_points = []
#     membrane_resnames = set()
#     for model in membrane_structure:
#         for chain in model:
#             for residue in chain:
#                 resname = residue.get_resname()
#                 if resname == "HOH":
#                     continue
#                 membrane_resnames.add(resname)
#                 for atom in residue.get_atoms():
#                     coord = atom.get_coord()
#                     membrane_points.append(coord)
#                     atom_name = atom.get_name().strip()
#                     if atom_name.startswith("P"):
#                         phosphate_points.append(coord)

#     if len(membrane_points) < 3:
#         raise ValueError("Not enough membrane atoms to define a plane.")

#     membrane_points = np.array(membrane_points)
#     phosphate_points = np.array(phosphate_points) if phosphate_points else None
#     plane_points = phosphate_points if phosphate_points is not None and len(phosphate_points) >= 3 else membrane_points

#     membrane_plane_center, membrane_plane_normal = _get_plane_from_points(plane_points)
#     membrane_axis = None
#     try:
#         centered = plane_points - membrane_plane_center
#         _, _, vh = np.linalg.svd(centered, full_matrices=False)
#         membrane_axis = vh[0]
#     except Exception:
#         membrane_axis = None

#     protein_inplane_axis = None
#     try:
#         chain_centers = []
#         for model in protein_structure:
#             for chain in model:
#                 chain_points = []
#                 for residue in chain:
#                     if not PDB.is_aa(residue):
#                         continue
#                     resseq = residue.id[1]
#                     if residue_start <= resseq <= residue_end and "CA" in residue:
#                         chain_points.append(residue["CA"].get_coord())
#                 if chain_points:
#                     chain_centers.append(np.mean(chain_points, axis=0))
#         if len(chain_centers) >= 2:
#             vec = chain_centers[1] - chain_centers[0]
#             vec = vec - protein_plane_normal * np.dot(vec, protein_plane_normal)
#             if np.linalg.norm(vec) > 1e-6:
#                 protein_inplane_axis = vec / np.linalg.norm(vec)
#     except Exception:
#         protein_inplane_axis = None

#     if rotate_membrane:
#         rotation = _rotation_matrix_from_vectors(membrane_plane_normal, protein_plane_normal)
#         _transform_structure(
#             membrane_structure,
#             rotation=rotation,
#             translation=np.zeros(3),
#             center=membrane_plane_center,
#         )
#         membrane_plane_normal = rotation.dot(membrane_plane_normal)
#         if membrane_axis is not None:
#             membrane_axis = rotation.dot(membrane_axis)

#     if inplane_align and membrane_axis is not None:
#         target_axis = protein_inplane_axis
#         if target_axis is None and protein_axis is not None:
#             target_axis = protein_axis - protein_plane_normal * np.dot(protein_axis, protein_plane_normal)
#             if np.linalg.norm(target_axis) > 1e-6:
#                 target_axis = target_axis / np.linalg.norm(target_axis)
#         if target_axis is None:
#             target_axis = None

#         membrane_proj = membrane_axis - membrane_plane_normal * np.dot(membrane_axis, membrane_plane_normal)
#         if target_axis is not None and np.linalg.norm(membrane_proj) > 1e-6:
#             membrane_proj = membrane_proj / np.linalg.norm(membrane_proj)
#             cross_val = np.cross(membrane_proj, target_axis)
#             sign = np.sign(np.dot(cross_val, membrane_plane_normal))
#             angle = np.arccos(np.clip(np.dot(membrane_proj, target_axis), -1.0, 1.0))
#             angle = angle * sign
#             if inplane_rotation_deg is not None:
#                 angle = np.deg2rad(inplane_rotation_deg)
#             if not np.isclose(angle, 0.0):
#                 rot_inplane = _rotation_matrix_axis_angle(membrane_plane_normal, angle)
#                 _transform_structure(
#                     membrane_structure,
#                     rotation=rot_inplane,
#                     translation=np.zeros(3),
#                     center=membrane_plane_center,
#                 )

#     rotated_points = []
#     for model in membrane_structure:
#         for chain in model:
#             for residue in chain:
#                 if residue.get_resname() == "HOH":
#                     continue
#                 for atom in residue.get_atoms():
#                     rotated_points.append(atom.get_coord())
#     rotated_points = np.array(rotated_points)
#     rotated_center = rotated_points.mean(axis=0)

#     translation = protein_plane_center - rotated_center
#     membrane_structure.transform(np.eye(3), translation)

#     if output_membrane_pdb_path:
#         io = PDBIO()
#         io.set_structure(membrane_structure)
#         io.save(output_membrane_pdb_path)

#     return {
#         "membrane_structure": membrane_structure,
#         "membrane_resnames": sorted(membrane_resnames),
#         "protein_plane_center": protein_plane_center,
#         "protein_plane_normal": protein_plane_normal,
#         "output_membrane_pdb_path": output_membrane_pdb_path,
#     }

def mutations_to_canonical(mutations, h3_map):
    """
    Convert mutation labels from sequence numbering to canonical H3 numbering.
    
    Args:
        mutations (list[str]): List of mutations with sequence numbering (e.g., ['A145K', 'G158E']).
        h3_map (dict): Mapping from 0-based sequence indices to canonical labels 
                       (from create_h3_numbering_map).
    
    Returns:
        list[str]: Mutations with canonical numbering (e.g., ['A158AK', 'G171E', 'HA2:A1T']).
    """
    canonical_muts = []
    
    for mut in mutations:
        # Extract components: original AA, position, new AA
        match = re.search(r'([A-Z*-])(\d+)([A-Z*-])', mut)
        if match:
            orig_aa = match.group(1)
            seq_pos_1based = int(match.group(2))
            new_aa = match.group(3)
            
            # Convert to 0-based index
            seq_pos_0based = seq_pos_1based - 1
            
            # Look up canonical label
            if seq_pos_0based in h3_map:
                canonical_label = h3_map[seq_pos_0based]
                
                # Handle HA2 labels specially
                if canonical_label.startswith('HA2:'):
                    # Format: HA2:S49N (not SHA2:49N)
                    # Extract the number part from HA2:49
                    position = canonical_label.split(':')[1]
                    canonical_mut = f"HA2:{orig_aa}{position}{new_aa}"
                else:
                    # Regular format: A158AK
                    canonical_mut = f"{orig_aa}{canonical_label}{new_aa}"
                
                canonical_muts.append(canonical_mut)
            else:
                # If not in map, keep original
                canonical_muts.append(mut)
        else:
            # If pattern doesn't match, keep original
            canonical_muts.append(mut)
    
    return canonical_muts

def create_h3_numbering_map(query_input, reference_sequence, signal_peptide_length=16, 
                            open_gap_score=-10, extend_gap_score=-0.5, HA2_start=None):
    """
    Creates a dictionary mapping sequence positions to canonical H3 numbering.
    Accepts Biopython SeqRecord objects or strings as input.

    Args:
        query_input (SeqRecord or str): The query sequence object from Biopython 
                                        (e.g., from SeqIO.parse) or a raw sequence string.
        reference_sequence (str): The reference sequence string (e.g., Aichi/68).
        signal_peptide_length (int): Length of signal peptide to skip (16 for Aichi).
        open_gap_score (float): Gap opening penalty.
        extend_gap_score (float): Gap extension penalty.
        HA2_start (int, optional): 1-based REFERENCE position where HA2 begins. When h3_counter
                                   reaches this value, positions will be labeled as 'HA2:1', 'HA2:2', etc.

    Returns:
        dict: {query_position (0-based): 'H3_Position_Label'} mapping positions in the 
              query sequence to canonical H3 numbering based on the reference.
    """
    
    # 1. Normalise Inputs
    # Check if query is a Biopython SeqRecord and extract the string if so
    if hasattr(query_input, 'seq'):
        query_seq_str = str(query_input.seq)
    else:
        query_seq_str = str(query_input)
        
    ref_seq_str = str(reference_sequence)

    # 2. Configure Aligner (Modern Biopython)
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score
    
    # Perform alignment
    alignments = aligner.align(ref_seq_str, query_seq_str)
    best_alignment = alignments[0]
    
    # Extract aligned strings (includes dashes)
    # pattern: target is ref, query is input sequence
    try:
        aligned_ref = best_alignment.target
        aligned_query = best_alignment.query
    except AttributeError:
        aligned_ref = best_alignment[0]
        aligned_query = best_alignment[1]

    # 3. Build the Mapping
    mapping_dict = {}
    
    # Counters
    # H3 numbering usually starts at 1 after the signal peptide. 
    # Positions in the SP are negative or labelled SP.
    h3_counter = 1 - signal_peptide_length 
    query_pos = 0  # Track position in ungapped query sequence (0-based)
    ha2_counter = 1  # Counter for HA2 region
    
    for ref_res, query_res in zip(aligned_ref, aligned_query):
        
        # We only care about mapping residues that actually exist in the query
        if query_res == '-':
            # If query has a gap, we just advance the reference counter (if ref has no gap)
            if ref_res != '-':
                h3_counter += 1
            continue

        # Save current query index (0-based)
        current_query_pos = query_pos
        query_pos += 1
        
        # Scenario A: Reference has a residue (Match or Mismatch)
        # We assign the current H3 number.
        if ref_res != '-':
            # Check if we're entering HA2 region (based on reference position)
            if HA2_start is not None and h3_counter >= HA2_start:
                # We're in HA2 region - use HA2 numbering
                label = f"HA2:{ha2_counter}"
                mapping_dict[current_query_pos] = label
                ha2_counter += 1
                h3_counter += 1
            elif h3_counter < 1:
                # Signal peptide region
                label = f"SP{h3_counter}"
                mapping_dict[current_query_pos] = label
                h3_counter += 1
            else:
                # Mature peptide region (HA1)
                label = str(h3_counter)
                mapping_dict[current_query_pos] = label
                h3_counter += 1
            
        # Scenario B: Reference has a gap (Insertion in Query)
        # We must assign an insertion code (e.g., 158A, 158B) based on the *previous* H3 number.
        else:
            if mapping_dict:
                # Retrieve the last assigned label
                last_label = list(mapping_dict.values())[-1]
                
                # Handle different label formats
                if last_label.startswith('HA2:'):
                    # HA2 insertion: HA2:49 -> HA2:49A, HA2:49B, etc.
                    base = last_label.rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                elif last_label.startswith('SP'):
                    # Signal peptide insertion: SP-15 -> SP-15A, SP-15B, etc.
                    base = last_label.rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                else:
                    # Regular HA1 insertion: 158 -> 158A, 158B, etc.
                    base = last_label.rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                
                # Calculate how deep into the insertion we are
                # Count how many times this base has appeared recently
                insertion_count = sum(1 for v in mapping_dict.values() 
                                    if v.startswith(base) and v != base)
                
                # Generate suffix: 0->A, 1->B, etc.
                suffix = chr(ord('A') + insertion_count)
                label = f"{base}{suffix}"
                mapping_dict[current_query_pos] = label
            else:
                # Edge case: Insertion at the very start of the sequence before any reference alignment
                mapping_dict[current_query_pos] = "N-term-Insert"

    return mapping_dict


def _tag_output_name(filename: str, test_mode: bool = False, diversity_pattern_tag: str = "", output_tag: str = "") -> str:
    stem, ext = os.path.splitext(filename)
    if test_mode:
        return f"test_{stem}_{diversity_pattern_tag}{ext}"
    return f"{stem}_{output_tag}{ext}"


def _is_probably_nucleotide(seq: str) -> bool:
    cleaned = seq.replace("-", "").replace(".", "").upper()
    if not cleaned:
        return False
    nuc_chars = set("ACGTUN")
    frac = sum(1 for char in cleaned if char in nuc_chars) / len(cleaned)
    return frac >= 0.95


def _translate_nt_to_protein(seq: str) -> str:
    cleaned = seq.replace("-", "").replace(".", "").upper().replace("U", "T")
    trimmed = cleaned[: (len(cleaned) // 3) * 3]
    if not trimmed:
        return ""
    return str(Seq(trimmed).translate(to_stop=False)).replace("*", "")


def parse_lineage_references(cluster_fasta: str, lineage_alias: Optional[dict] = None):
    if lineage_alias is None:
        lineage_alias = {}
    refs = {}
    for record in SeqIO.parse(cluster_fasta, "fasta"):
        header = record.id.strip()
        lineage_raw = header.split("|")[-1] if "|" in header else header
        lineage = lineage_alias.get(lineage_raw, lineage_raw)

        raw_seq = str(record.seq)
        nt_seq = raw_seq.replace("-", "").replace(".", "").upper().replace("U", "T")
        if _is_probably_nucleotide(raw_seq):
            protein_seq = _translate_nt_to_protein(raw_seq)
        else:
            protein_seq = raw_seq.replace("-", "")

        if lineage not in refs:
            refs[lineage] = {
                "header": header,
                "lineage": lineage,
                "nucleotide": nt_seq,
                "protein": protein_seq,
            }
    return refs


def load_lineage_diversity_fastas(diversity_dir: str, file_pattern: str):
    lineage_files = {}
    pattern = os.path.join(diversity_dir, file_pattern)
    for path in glob.glob(pattern):
        filename = os.path.basename(path)
        m = re.match(r"H3N2_(.+)_(max[^.]*)\.fasta$", filename)
        if not m:
            continue
        lineage_key = m.group(1)
        diversity_tag = m.group(2)
        records = list(SeqIO.parse(path, "fasta"))
        lineage_files[lineage_key] = {
            "path": path,
            "diversity_tag": diversity_tag,
            "records": records,
        }
    return lineage_files


def compute_lineage_mutation_profile(reference_nt: str, reference_protein: str, aa_to_codons: dict, codon_mutation_df: pd.DataFrame):
    profile = pd.DataFrame(
        0.0,
        index=list(aa_to_codons.keys()),
        columns=list(range(1, len(reference_protein) + 1)),
    )

    for pos1 in range(1, len(reference_protein) + 1):
        codon = reference_nt[(pos1 - 1) * 3 : pos1 * 3]
        if len(codon) != 3 or codon not in codon_mutation_df.index:
            continue
        for target_aa, target_codons in aa_to_codons.items():
            total = 0.0
            for tc in target_codons:
                if tc in codon_mutation_df.columns:
                    total += float(codon_mutation_df.loc[codon, tc])
            profile.loc[target_aa, pos1] = total

    return profile


def compute_observed_diversity_profile(records, reference_protein: str, aa_to_codons: dict):
    aa_order = sorted(list(aa_to_codons.keys()))
    n_pos = len(reference_protein)

    counts = pd.DataFrame(0, index=aa_order, columns=list(range(1, n_pos + 1)))
    valid_depth = pd.Series(0, index=list(range(1, n_pos + 1)), dtype=float)

    for record in records:
        seq = str(record.seq)
        seq = seq[:n_pos] if len(seq) >= n_pos else seq + ("-" * (n_pos - len(seq)))
        for pos1 in range(1, n_pos + 1):
            aa = seq[pos1 - 1]
            if aa == "-":
                continue
            if aa in counts.index:
                counts.loc[aa, pos1] += 1
            valid_depth[pos1] += 1

    freqs = counts.copy().astype(float)
    for pos1 in freqs.columns:
        depth = valid_depth[pos1]
        if depth > 0:
            freqs[pos1] = freqs[pos1] / depth
        else:
            freqs[pos1] = 0.0

    return freqs, valid_depth


def _build_lineage_consensus_and_column_map(records, aa_to_codons: dict, ignore_chars: set):
    if len(records) == 0:
        return "", [], 0

    aligned_sequences = [str(record.seq).upper() for record in records]
    aln_len = max(len(seq) for seq in aligned_sequences)
    seq_array = np.array([list(seq.ljust(aln_len, "-")) for seq in aligned_sequences])

    aa_order = sorted(list(aa_to_codons.keys()))
    valid_aas = set(aa_order)

    consensus_chars = []
    consensus_to_alignment_col = []

    for col_idx in range(aln_len):
        col_vals = seq_array[:, col_idx]
        valid_vals = [aa for aa in col_vals if (aa not in ignore_chars and aa in valid_aas)]
        if len(valid_vals) == 0:
            continue
        residues, counts = np.unique(valid_vals, return_counts=True)
        consensus_aa = residues[np.argmax(counts)]
        consensus_chars.append(consensus_aa)
        consensus_to_alignment_col.append(col_idx + 1)  # 1-based

    return "".join(consensus_chars), consensus_to_alignment_col, aln_len


def build_reference_to_alignment_column_map(reference_protein: str, records, aa_to_codons: dict, ignore_chars: set):
    consensus_seq, consensus_to_alignment_col, aln_len = _build_lineage_consensus_and_column_map(records, aa_to_codons, ignore_chars)
    if not consensus_seq:
        return {}, aln_len, 0

    alignments = pairwise2.align.globalms(
        reference_protein,
        consensus_seq,
        2.0,
        -1.0,
        -10.0,
        -0.5,
        one_alignment_only=True,
    )
    if len(alignments) == 0:
        return {}, aln_len, 0

    ref_aln, cons_aln = alignments[0].seqA, alignments[0].seqB

    ref_pos = 0
    cons_pos = 0
    mapping = {}
    matched_pairs = 0
    for ref_char, cons_char in zip(ref_aln, cons_aln):
        if ref_char != "-":
            ref_pos += 1
        if cons_char != "-":
            cons_pos += 1

        if ref_char != "-" and cons_char != "-":
            if 1 <= cons_pos <= len(consensus_to_alignment_col):
                mapping[ref_pos] = consensus_to_alignment_col[cons_pos - 1]
                matched_pairs += 1

    return mapping, aln_len, matched_pairs


def compute_observed_diversity_profile_fast(records, reference_protein: str, ref_to_aln_col: dict, aln_len: int, aa_to_codons: dict, ignore_chars: set):
    n_pos = len(reference_protein)
    aa_order = sorted(list(aa_to_codons.keys()))
    counts = pd.DataFrame(0.0, index=aa_order, columns=list(range(1, n_pos + 1)))
    valid_depth = pd.Series(0.0, index=list(range(1, n_pos + 1)), dtype=float)

    if len(records) == 0 or aln_len <= 0:
        return counts, valid_depth, {
            "mapped_sites": 0,
            "compared_sites": 0,
            "differing_sites": 0,
            "fixed_differing_sites": 0,
            "alignment_length": int(aln_len),
        }

    seq_array = np.array([list(str(record.seq).upper().ljust(aln_len, "-")) for record in records])

    differing_sites = 0
    fixed_differing_sites = 0
    compared_sites = 0
    valid_aas = set(aa_order)

    for pos1 in range(1, n_pos + 1):
        aln_col = ref_to_aln_col.get(pos1)
        if aln_col is None or aln_col < 1 or aln_col > aln_len:
            continue

        residues = seq_array[:, aln_col - 1]
        residues = np.array([aa for aa in residues if aa not in ignore_chars and aa in valid_aas])
        depth = int(len(residues))
        valid_depth[pos1] = depth
        if depth == 0:
            continue

        compared_sites += 1
        ref_aa = reference_protein[pos1 - 1]
        has_any_difference = bool(np.any(residues != ref_aa))
        has_fixed_difference = bool(np.all(residues != ref_aa))
        if has_any_difference:
            differing_sites += 1
        if has_fixed_difference:
            fixed_differing_sites += 1

        uniq, cnt = np.unique(residues, return_counts=True)
        for aa, c in zip(uniq, cnt):
            counts.loc[aa, pos1] = float(c)

    freqs = counts.copy()
    for pos1 in freqs.columns:
        depth = valid_depth[pos1]
        if depth > 0:
            freqs[pos1] = freqs[pos1] / depth
        else:
            freqs[pos1] = 0.0

    stats = {
        "mapped_sites": int(len(ref_to_aln_col)),
        "compared_sites": int(compared_sites),
        "differing_sites": int(differing_sites),
        "fixed_differing_sites": int(fixed_differing_sites),
        "alignment_length": int(aln_len),
    }
    return freqs, valid_depth, stats

def softmax_from_log_scores(log_scores: np.ndarray) -> np.ndarray:
    shifted = log_scores - np.nanmax(log_scores)
    ex = np.exp(shifted)
    denom = np.nansum(ex)
    if denom <= 0:
        return np.zeros_like(log_scores)
    return ex / denom


def _extract_corr_pvalue(result):
    """Return (correlation, pvalue) from scipy result object or tuple."""
    try:
        return float(result[0]), float(result[1])
    except Exception:
        corr = getattr(result, "correlation", getattr(result, "statistic", np.nan))
        pval = getattr(result, "pvalue", np.nan)
        return float(corr), float(pval)


def _evaluate_single_alpha(alpha: float, base_df: pd.DataFrame, pseudocount: float = 1e-6) -> dict:
    working = base_df.copy()
    working["combined_log_score"] = working["log_plm"] + alpha * working["log_mut"]

    global_spearman = spearmanr(working["combined_log_score"], working["obs_freq"])
    global_pearson = pearsonr(working["combined_log_score"], working["obs_freq"])
    sp_r, sp_p = _extract_corr_pvalue(global_spearman)
    pr_r, pr_p = _extract_corr_pvalue(global_pearson)

    top_frac = 0.05
    n_top = max(1, int(len(working) * top_frac))
    ranked = working.sort_values("combined_log_score", ascending=False)
    top_hits = ranked.head(n_top)
    baseline_prevalence = float(working["obs_present"].mean()) if len(working) > 0 else np.nan
    top_prevalence = float(top_hits["obs_present"].mean()) if len(top_hits) > 0 else np.nan
    top_enrichment = top_prevalence / baseline_prevalence if baseline_prevalence and baseline_prevalence > 0 else np.nan

    site_view = (
        working.groupby(["lineage", "position", "ref_aa"], as_index=False)
        .agg(
            site_pred_score=("combined_log_score", "max"),
            site_obs_burden=("obs_freq", "sum"),
            site_mutated=("obs_present", "max"),
        )
    )

    site_spearman = (
        spearmanr(site_view["site_pred_score"], site_view["site_obs_burden"])
        if len(site_view) > 1
        else (np.nan, np.nan)
    )
    site_sp_r, site_sp_p = _extract_corr_pvalue(site_spearman)

    site_top_frac = 0.10
    n_site_top = max(1, int(len(site_view) * site_top_frac)) if len(site_view) > 0 else 0
    top_site_hits = site_view.sort_values("site_pred_score", ascending=False).head(n_site_top) if n_site_top > 0 else site_view.head(0)
    site_top_precision = float(top_site_hits["site_mutated"].mean()) if len(top_site_hits) > 0 else np.nan
    site_baseline = float(site_view["site_mutated"].mean()) if len(site_view) > 0 else np.nan
    site_top_enrichment = site_top_precision / site_baseline if site_baseline and site_baseline > 0 else np.nan

    site_nlls = []
    site_rhos = []
    grouped = working.groupby(["lineage", "position", "ref_aa"], sort=False)
    for (_, _, _), site_df in grouped:
        obs_vec = site_df["obs_freq"].to_numpy(dtype=float)
        score_vec = site_df["combined_log_score"].to_numpy(dtype=float)
        if np.nansum(obs_vec) <= 0:
            continue

        obs_norm = obs_vec / np.nansum(obs_vec)
        pred_prob = softmax_from_log_scores(score_vec)
        nll = -np.nansum(obs_norm * np.log(pred_prob.clip(min=pseudocount)))
        site_nlls.append(float(nll))

        if np.nanstd(obs_vec) > 0 and np.nanstd(score_vec) > 0:
            rho_result = spearmanr(score_vec, obs_vec)
            rho, _ = _extract_corr_pvalue(rho_result)
            if np.isfinite(rho):
                site_rhos.append(float(rho))

    return {
        "alpha": float(alpha),
        # Method B: mutation-level flattened ranking (19xN entries)
        "mut_flat_global_spearman_r": float(sp_r) if np.isfinite(sp_r) else np.nan,
        "mut_flat_global_spearman_p": float(sp_p) if np.isfinite(sp_p) else np.nan,
        "mut_flat_global_pearson_r": float(pr_r) if np.isfinite(pr_r) else np.nan,
        "mut_flat_global_pearson_p": float(pr_p) if np.isfinite(pr_p) else np.nan,
        "mut_flat_top5pct_enrichment": float(top_enrichment) if np.isfinite(top_enrichment) else np.nan,
        "mut_flat_mean_site_nll": float(np.mean(site_nlls)) if len(site_nlls) > 0 else np.nan,
        "mut_flat_median_site_spearman": float(np.median(site_rhos)) if len(site_rhos) > 0 else np.nan,
        # Method A: site-level ranking (N entries)
        "site_rank_spearman_r": float(site_sp_r) if np.isfinite(site_sp_r) else np.nan,
        "site_rank_spearman_p": float(site_sp_p) if np.isfinite(site_sp_p) else np.nan,
        "site_top10pct_mutated_precision": float(site_top_precision) if np.isfinite(site_top_precision) else np.nan,
        "site_top10pct_mutated_enrichment": float(site_top_enrichment) if np.isfinite(site_top_enrichment) else np.nan,
        "n_sites_used": int(len(site_nlls)),
    }


def evaluate_alpha_sweep(
    combined_df: pd.DataFrame,
    alpha_grid: np.ndarray,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    alpha_sweep_min_grid: int = 10,
    pseudocount: float = 1e-6,
) -> pd.DataFrame:
    base_df = combined_df.copy()
    base_df["log_plm"] = np.log(base_df["plm_prob"].clip(lower=pseudocount))
    base_df["log_mut"] = np.log(base_df["mut_prob"].clip(lower=pseudocount))

    alpha_values = [float(alpha) for alpha in alpha_grid]
    if parallel and len(alpha_values) >= alpha_sweep_min_grid:
        workers = max_workers if max_workers is not None else min(len(alpha_values), max(1, os.cpu_count() or 1))
        # Note: You'll need ThreadPoolExecutor imported if parallel=True is used
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as executor:
            alpha_results = list(executor.map(lambda alpha: _evaluate_single_alpha(alpha, base_df, pseudocount), alpha_values))
    else:
        alpha_results = [_evaluate_single_alpha(alpha, base_df, pseudocount) for alpha in alpha_values]

    return pd.DataFrame(alpha_results).sort_values("alpha")


## Convenience helpers used by notebooks ---------------------------------------
def load_plm_probability_matrix(path):
    """Load a PLM probability matrix CSV in the flexible formats used by notebooks.

    Returns the raw dataframe as read with index_col=0, header=None when possible.
    """
    try:
        df = pd.read_csv(path, index_col=0, header=None)
    except Exception:
        df = pd.read_csv(path, index_col=0)
    return df


def calculate_mutational_accessibility(probability_matrix, codon_mutation_df, aa_to_codons, reference_nt=None):
    """Given a PLM probability matrix (as read in the notebooks), produce a
    codon->AA mutational probability matrix and combined PLM×mutational matrices.

    Parameters
    - probability_matrix: DataFrame (often read with header=None, index_col=0)
    - codon_mutation_df: DataFrame 64x64 of codon->codon probabilities
    - aa_to_codons: dict mapping amino-acid letter -> list of codons
    - reference_nt: optional nucleotide reference sequence (string). If provided
      it is used to map columns to codons; otherwise the function will attempt
      to infer codons from the probability_matrix header row (first row).

    Returns dict with keys: mutational_prob_matrix (DataFrame), plm_matrix (DataFrame),
    combined_prob_matrix (DataFrame), combined_prob_sqrt_matrix (DataFrame)
    """
    # Detect header-row sequence (common notebook format: first row contains sequence)
    pm = probability_matrix.copy()
    plm_matrix = None
    # If first row contains non-numeric values and more rows exist, treat it as header
    first_row = pm.iloc[0, :]
    if first_row.apply(lambda x: isinstance(x, str)).any() and pm.shape[0] > 1:
        # numeric part starts at row 1
        try:
            plm_matrix = pm.iloc[1:, :].apply(pd.to_numeric, errors="coerce")
        except Exception:
            plm_matrix = pm.iloc[1:, :]
    else:
        # assume whole df is numeric PLM matrix
        plm_matrix = pm.apply(pd.to_numeric, errors="coerce")

    # If reference_nt provided, build codon list from it
    if reference_nt is not None:
        ref = reference_nt.upper().replace("U", "T").replace(" ", "")
        ncols = plm_matrix.shape[1]
        codons = []
        for i in range(ncols):
            start = i * 3
            cod = ref[start : start + 3]
            codons.append(cod)
    else:
        # try to infer codons from original probability_matrix header row
        header_seq = pm.iloc[0, :].astype(str).tolist()
        # if header_seq looks like a sequence string concatenated, try split by characters
        if len(header_seq) >= plm_matrix.shape[1] and all(len(s) == 1 for s in header_seq[: plm_matrix.shape[1]]):
            # join and take triplets
            seq = "".join(header_seq[: plm_matrix.shape[1]])
            codons = [seq[i * 3 : i * 3 + 3] for i in range(plm_matrix.shape[1])]
        else:
            # fallback: cannot determine codons; use placeholder 'NNN'
            codons = ["NNN"] * plm_matrix.shape[1]

    # Build mutational probability matrix: rows AA, columns positions (column labels copied from plm_matrix)
    aas = sorted(aa_to_codons.keys())
    mut_prob_df = pd.DataFrame(0.0, index=aas, columns=plm_matrix.columns)

    for col_idx, col_label in enumerate(plm_matrix.columns):
        cod = codons[col_idx] if col_idx < len(codons) else "NNN"
        if cod not in codon_mutation_df.index:
            # leave zeros
            continue
        for aa in aas:
            total = 0.0
            for tc in aa_to_codons.get(aa, []):
                if tc in codon_mutation_df.columns:
                    total += float(codon_mutation_df.loc[cod, tc])
            mut_prob_df.loc[aa, col_label] = total

    combined = plm_matrix.copy() * mut_prob_df
    combined_sqrt = plm_matrix.copy() * np.sqrt(mut_prob_df)

    return {
        "mutational_prob_matrix": mut_prob_df,
        "plm_matrix": plm_matrix,
        "combined_prob_matrix": combined,
        "combined_prob_sqrt_matrix": combined_sqrt,
    }


def plot_mutational_accessibility_heatmap(matrix_df, outpath=None, figsize=(14, 8), cmap="viridis"):
    """Plot and optionally save a heatmap for a mutational accessibility DataFrame.

    matrix_df: DataFrame (rows AA, columns positions)
    outpath: optional Path to save PNG
    """
    plt.figure(figsize=figsize)
    ax = sns.heatmap(matrix_df, cmap=cmap, cbar_kws={"label": "Probability"})
    ax.set_xlabel("Position")
    ax.set_ylabel("Amino Acid")
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(str(outpath), dpi=300)
    plt.show()

