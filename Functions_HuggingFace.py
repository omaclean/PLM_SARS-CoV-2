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
    return pd.concat(seq_list,columns=['Mutations','Sequence'])

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
    """Embed a protein sequence with a HuggingFace ESM model.

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
        # FIX: Use output_hidden_states instead of repr_layers
        results = model(batch_tokens, output_hidden_states=True)
    del batch_tokens

    # FIX: Handle Layer Indexing
    # HF hidden_states is a tuple: (embeddings, layer_1, ... layer_N)
    # Length is num_layers + 1.
    # If model_layers is too high (e.g. 33 for a 30-layer model), default to the last layer (-1).
    
    try:
        token_representation = results.hidden_states[model_layers][0]
    except IndexError:
        print(f"Warning: Requested layer {model_layers} is out of bounds for this model. Using last layer.")
        token_representation = results.hidden_states[-1][0]

    full_embedding = token_representation[1:batch_len - 1].cpu()
    base_mean_embedding = token_representation[1 : batch_len - 1].mean(0).cpu()

    # FIX: Access logits directly
    lsoftmax = torch.nn.LogSoftmax(dim=1)
    base_logits = lsoftmax((results.logits[0]).to(device="cpu"))
    
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
                                        "sequence_grammaticality":get_sequence_grammaticality(reference_protein,reference_logits,alphabet)
                                        
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
            
        #If no need for scores, sequences can be embedded as is with insertions, deletions truncations etc
        else:
            embeddings[coding_region_name][name] = process_protein_sequence(str(fasta.seq),model,model_layers,batch_converter,alphabet,device)
        i+=1
    return embeddings

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
  
def visualise_mutations_on_pdb(pdb_file, user_sequence, mutation_list, threshold_score=50):
    """
    Maps user-defined mutations onto a PDB structure (trimer friendly).
    
    Args:
        pdb_file (str): Path to .pdb file.
        user_sequence (str): The specific K lineage AA sequence string.
        mutation_list (list): List of strings e.g., ['A123T', 'N145K'].
                              Assumes 1-based indexing in the string.
        threshold_score (int): Minimum alignment score to consider a chain a "match".
    """
    
    # 1. Parse Mutation Indices from the list (e.g. 'A123T' -> 122 (0-based))
    # We assume the input list uses standard 1-based biological numbering
    sites_of_interest = []
    for mut in mutation_list:
        match = re.search(r'(\d+)', mut)
        if match:
            # Convert 1-based str index to 0-based list index
            sites_of_interest.append(int(match.group(1)) - 1)
    
    # 2. Parse PDB Structure
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_file)
    
    # Setup Aligner (Smith-Waterman Local)
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    # High gap penalties to prevent breaking the helix/sheet structure in alignment
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5

    # Dictionary to store hits: { (ChainID, ResidueID) : Color }
    # py3Dmol expects ResidueID as integer, but PDB might have insertion codes.
    # We will store raw PDB res numbers.
    residues_to_highlight = {}

    print(f"Processing PDB: {pdb_file}")
    
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

            # Align User Seq vs Chain Seq
            alignment = aligner.align(user_sequence, pdb_seq_str)[0]
            
            # Simple score check to see if this chain is the protein of interest (HA1)
            # and not the stem (HA2) or a nanobody
            if alignment.score < threshold_score:
                continue
                
            print(f"  -> Match found on Chain {chain.id} (Score: {alignment.score:.1f})")

            # 4. Map User Indices to PDB Residues using the Alignment
            # We iterate through the alignment trace
            # aligned arrays: [0] is query (user), [1] is target (pdb)
            
            # alignment.indices is a list of aligned indices for the target and query
            # But extracting exact matches requires walking the path.
            
            # Get the indices of the matches in both sequences
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
            
            for site in sites_of_interest:
                if site in user_to_pdb_array_map:
                    pdb_array_idx = user_to_pdb_array_map[site]
                    
                    # Retrieve the REAL PDB residue ID (handling numbering/insertion codes)
                    # pdb_residues is a list of PDB residue objects aligned with pdb_seq_str
                    try:
                        real_residue_id = pdb_residues[pdb_array_idx]
                        # real_residue_id example: (' ', 158, ' ') or (' ', 158, 'A')
                        
                        resi_num = real_residue_id[1]
                        # py3Dmol handles insertion codes via specific selection if needed, 
                        # but usually just ID is enough for broad visualisation. 
                        # If you have insertion codes (e.g. 144A), py3Dmol needs carefully formatted strings.
                        
                        # Store for plotting
                        if chain.id not in residues_to_highlight:
                            residues_to_highlight[chain.id] = []
                        residues_to_highlight[chain.id].append(resi_num)
                        
                    except IndexError:
                        pass

    # 5. Visualise with py3Dmol
    view = py3Dmol.view(width=800, height=600)
    
    # Use doAssembly=True to generate the biological assembly (e.g. trimer)
    view.addModel(open(pdb_file).read(), 'pdb', {'doAssembly': True})
    
    # Base style: Grey Cartoon
    view.setStyle({'cartoon': {'color': '#eeeeee'}})
    
    # Loop through our mapped hits and colour them
    count = 0
    for chain_id, res_nums in residues_to_highlight.items():
        # Remove duplicates
        res_nums = list(set(res_nums))
        count += len(res_nums)
        
        # Apply style to specific residues
        # When doAssembly=True, py3Dmol might create multiple models or rename chains.
        # If the assembly consists of copies of the chains we parsed (e.g. A, B),
        # specifying {'chain': chain_id} will target that chain in ALL models if they preserve the ID.
        # If they are renamed, we might miss them. 
        # However, usually for homo-oligomers or simple assemblies, keeping the chain ID is safer 
        # than applying to all chains (which might highlight the wrong protein in a hetero-complex).
        # We assume here that the assembly preserves chain IDs across models or we only care about the primary chains.
        
        # To be more robust for multimers where we want to highlight ALL copies of the matching chain:
        # We can try to apply the style to the specific chain ID.
        
        view.addStyle(
            {'chain': chain_id, 'resi': res_nums},
            {'cartoon': {'color': '#FF4136'}} # Bright Red for backbone
        )
        view.addStyle(
            {'chain': chain_id, 'resi': res_nums},
            {'stick': {'colorscheme': 'redCarbon'}} # Sticks for sidechains
        )
        view.addStyle(
             {'chain': chain_id, 'resi': res_nums},
             {'surface': {'opacity':0.5, 'color': '#FF4136'}} # Transparent surface
        )

    view.zoomTo()
    print(f"Mapped {count} mutation sites across the structure (showing biological assembly).")
    return view
