"""

PRIDICT2.0 pegRNA prediction script including calculation of features.
Additionally, nicking guides (including DeepSpCas9 score from Kim et al. 2019)
and primers for NGS amplicon (based on Primer3 from Untergasser et al. 2012) 
are designed in separate output file.

To check editability of your locus based on chromatin context, 
check out ePRIDICT (epigenetic based PRIme editing efficiency preDICTion),
available from https://github.com/Schwank-Lab/epridict.

"""


### set parameters first ###

# set filename of batchfile located in same folder as this python script
batchfile = 'batchfile_template.csv'

# set all PBS lengths which should be predicted; more PBS lengths leads to longer prediction time; default 7-15 bp
PBSlengthrange = range(7,16)


# set all RToverhang lengths which should be predicted; more RToverhang lengths leads to longer prediction time; default 3-19 bp
RToverhanglengthrange = range(3,20)

# set maximum distance of edit to PAM; longer distance leads to longer prediction time; default 25
windowsize_max=25

# Define whether DeepSpCas9 prediction should be performed for nicking guides; default True
# Changing value to False will accelerate prediction by adding dummy prediction values to nicking guides.
# Does not influence predictions performed by PRIDICT model (only affects nicking guides)
deepcas9_trigger=True

### end of parameter delcaration ###


import re
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt

import pandas as pd
import os
import time
import torch.multiprocessing as mp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mp.set_start_method("spawn", force=True)
import argparse
import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", "Function deprecated please use \"design_primers\" instead", category=UserWarning)
import primer3

from pridict.pridictv2.utilities import *
from pridict.pridictv2.dataset import *
from pridict.pridictv2.predict_outcomedistrib import *
from trained_models.DeepCas9_TestCode import runprediction

    

def primesequenceparsing(sequence: str) -> object:
    """
    Function which takes target sequence with desired edit as input and 
    editing characteristics as output. Edit within brackets () and original
    equence separated by backslash from edited sequence: (A/G) == A to G mutation.
    Placeholder for deletions and insertions is '-'.

    Parameters
    ----------
    sequence : str
        Target sequence with desired edit in brackets ().


    """
    
    sequence = sequence.replace('\n','')  # remove any spaces or linebreaks in input
    sequence = sequence.replace(' ','')
    sequence = sequence.upper()
    if sequence.count('(') != 1:
        print(sequence)
        print('More or less than one bracket found in sequence! Please check your input sequence.')
        raise ValueError

    five_prime_seq = sequence.split('(')[0]
    three_prime_seq = sequence.split(')')[1]

    sequence_set = set(sequence)
    if '/' in sequence_set:
        original_base = sequence.split('/')[0].split('(')[1]
        edited_base = sequence.split('/')[1].split(')')[0]

        # edit flanking bases should *not* be included in the brackets
        if (original_base[0] == edited_base[0]) or (original_base[-1] == edited_base[-1]):
            print(sequence)
            print('Flanking bases should not be included in brackets! Please check your input sequence.')
            raise ValueError
    elif '+' in sequence_set:  #insertion
        original_base = '-'
        edited_base = sequence.split('+')[1].split(')')[0]
    elif '-' in sequence_set:  #deletion
        original_base = sequence.split('-')[1].split(')')[0]
        edited_base = '-'

    # ignore "-" in final sequences (deletions or insertions)
    if original_base == '-':
        original_seq = five_prime_seq + three_prime_seq
        if edited_base != '-':
            mutation_type = 'Insertion'
            correction_length = len(edited_base)
        else:
            print(sequence)
            raise ValueError
    else:
        original_seq = five_prime_seq + original_base + three_prime_seq
        if edited_base == '-':
            mutation_type = 'Deletion'
            correction_length = len(original_base)
        elif len(original_base) == 1 and len(edited_base) == 1:
            if isDNA(original_base) and isDNA(edited_base):  # check if only AGCT is in bases
                mutation_type = '1bpReplacement'
                correction_length = len(original_base)
            else:
                print(sequence)
                print('Non DNA bases found in sequence! Please check your input sequence.')
                raise ValueError
        elif len(original_base) > 1 or len(edited_base) > 1:
            if isDNA(original_base) and isDNA(edited_base):  # check if only AGCT is in bases
                mutation_type = 'MultibpReplacement'
                if len(original_base) == len(
                        edited_base):  # only calculate correction length if replacement does not contain insertion/deletion
                    correction_length = len(original_base)
                else:
                    print(sequence)
                    print('Only 1bp replacements or replacements of equal length (before edit/after edit) are supported! Please check your input sequence.')
                    raise ValueError
            else:
                print(sequence)
                print('Non DNA bases found in sequence! Please check your input sequence.')
                raise ValueError

    if edited_base == '-':
        edited_seq = five_prime_seq + three_prime_seq
    else:
        edited_seq = five_prime_seq + edited_base.lower() + three_prime_seq

    if isDNA(edited_seq) and isDNA(original_seq):  # check whether sequences only contain AGCT
        pass
    else:
        raise ValueError

    basebefore_temp = five_prime_seq[
                      -1:]  # base before the edit, could be changed with baseafter_temp if Rv strand is targeted (therefore the "temp" attribute)
    baseafter_temp = three_prime_seq[:1]  # base after the edit

    editposition_left = len(five_prime_seq)
    editposition_right = len(three_prime_seq)
    return original_base, edited_base, original_seq, edited_seq, editposition_left, editposition_right, mutation_type, correction_length, basebefore_temp, baseafter_temp


def editorcharacteristics(editor):
    if editor == 'PE2-NGG':
        PAM = '(?=GG)'
        numberN = 1
        PAM_length = len(PAM) - 4 + numberN # 3 in this case
        variant = 'PE2-NGG'
        protospacerlength = 19
        PAM_side = 'right'
        primescaffoldseq = 'GTTTCAGAGCTATGCTGGAAACAGCATAGCAAGTTGAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC'
        # primescaffoldseq = 'GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC'
    return PAM, numberN, variant, protospacerlength, PAM_side, primescaffoldseq, PAM_length


def isDNA(sequence):
    """ Check whether sequence contains only DNA bases. """
    onlyDNA = True
    diff_set = set(sequence) - set('ACTGatgc')
    if diff_set:
        onlyDNA = False
        print('Non-DNA bases detected. Please use ATGC.')
        print(sequence)
        raise ValueError
    return onlyDNA

def melting_temperature(protospacer, extension, RT, RToverhang, PBS, original_base, edited_base):
    """"Calculate melting temperature for different sequences."""
    protospacermt = mt.Tm_Wallace(Seq(protospacer))
    extensionmt = mt.Tm_Wallace(Seq(extension))
    RTmt = mt.Tm_Wallace(Seq(RT))
    RToverhangmt = mt.Tm_Wallace(Seq(RToverhang))
    PBSmt = mt.Tm_Wallace(Seq(PBS))

    if original_base == '-':
        original_base_mt = 0
        original_base_mt_nan = 1
    else:
        original_base_mt = mt.Tm_Wallace(Seq(original_base))
        original_base_mt_nan = 0

    if edited_base == '-':
        edited_base_mt = 0
        edited_base_mt_nan = 1
    else:
        edited_base_mt = mt.Tm_Wallace(Seq(edited_base))
        edited_base_mt_nan = 0

    return protospacermt, extensionmt, RTmt, RToverhangmt, PBSmt, original_base_mt, edited_base_mt, original_base_mt_nan, edited_base_mt_nan


def RToverhangmatches(RToverhang, edited_seq, RToverhangstartposition, RTlengthoverhang):
    """"Counts whether RToverhang matches up to 15bp downstream of designated position in edited_seq (e.g. due to repetitive motivs) which would prevent editing of certain deletions or insertions (e.g. A(A/-)AAA with 3bp RT overhang)"""
    RToverhangmatchcount = occurrences(
        edited_seq[RToverhangstartposition:RToverhangstartposition + RTlengthoverhang + 15], RToverhang)
    return RToverhangmatchcount

def multideepeditpositionfunc(originalbases, editedbases, deepeditposition):
    '''Gives position when having multiple edits; e.g. CAG to GAT would be [0,2]. But we want to add the absolute not relative position, therefore we add the deepeditposition (based on wide_initial_target) on top.'''
    multideepeditlist = []
    for i in range(len(originalbases)):
        if originalbases[i] != editedbases[i]:
            multideepeditlist.append(deepeditposition+i)
    return multideepeditlist

def occurrences(string, sub):
    """"Gives total count of substring in string including overlapping substrings."""
    count = start = 0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count += 1
        else:
            return count


def deepcas9(deepcas9seqlist):
    """Perform DeepCas9 prediction on 30bp stretches of protospacer + targetseq for each protospacer."""
    if deepcas9_trigger == True:
        usecase = 'commandline'
        deepcas9scorelist = runprediction(deepcas9seqlist, usecase)
        print('deepcas9 calculating...')
        deepcas9scorelist = [round(x, 2) for x in deepcas9scorelist]
        return deepcas9scorelist
    else:  # if deepcas9_trigger is set to false, just calculate dummy data for nicking guides.
        print('deepcas9 calculating...')
        deepcas9scorelist = [100.]*len(deepcas9seqlist)
        return deepcas9scorelist


def nickingguide(original_seq, PAMposition, protospacerlength):
    """Define nickingguide and corresponding 30bp stretch for DeepCas9."""
    nickprotospacer = original_seq[PAMposition - 1 - protospacerlength:PAMposition - 1]
    nickdeepcas9 = original_seq[PAMposition - 1 - protospacerlength - 4 - (20 - protospacerlength):PAMposition - 1 + 6]

    return nickprotospacer, nickdeepcas9

def load_pridict_model(run_ids=[0]):
    """construct and return PRIDICT model along with model files directory """
    models_lst_dict = {}  # Initialize a dictionary to hold lists of models keyed by model_id
    device = get_device(True, 0)
    wsize = 20
    repo_dir = os.path.join(os.path.abspath('./'))
    modellist = [
        ('PRIDICT1_1', 'pe_rnn_distribution_multidata', 'exp_2023-08-25_20-55-53'),
        ('PRIDICT1_2', 'pe_rnn_distribution_multidata', 'exp_2023-08-28_22-22-26')
    ]
    for model in modellist:
        models_lst = []  # Initialize models_lst for each model
        model_id, original_folder, mfolder = model

        prieml_model = PRIEML_Model(device, wsize=wsize, normalize='max', fdtype=torch.float32)
        
        for run_num in run_ids:
            model_dir = os.path.join(repo_dir, 'trained_models', model_id.lower(), mfolder, 'train_val', f'run_{run_num}')
            prieml_model.build_retrieve_models(model_dir)
            models_lst.append((prieml_model, model_dir))
        
        models_lst_dict[model_id] = models_lst  # Add to the dictionary

    return models_lst_dict

def deeppridict(pegdataframe, models_lst_dict):
    """Perform score prediction on dataframe of features based on RNN model.
    
    Args:
        pegdataframe: pandas DataFrame containing the processed sequence features
        models_lst: list of tuples of (pridict_model, model_run_dir)
    
    """
    all_avg_preds = {} 

    # setup the dataframe
    deepdfcols = ['wide_initial_target', 'wide_mutated_target', 'deepeditposition', 
                  'deepeditposition_lst', 'Correction_Type', 'Correction_Length', 
                  'protospacerlocation_only_initial', 'PBSlocation',
                  'RT_initial_location', 'RT_mutated_location',
                  'RToverhangmatches', 'RToverhanglength', 
                  'RTlength', 'PBSlength', 'RTmt', 'RToverhangmt','PBSmt','protospacermt',
                  'extensionmt','original_base_mt','edited_base_mt','original_base_mt_nan',
                  'edited_base_mt_nan']

    deepdf = pegdataframe[deepdfcols].copy()
    deepdf.insert(1, 'seq_id', range(len(deepdf)))
    deepdf['protospacerlocation_only_initial'] = deepdf['protospacerlocation_only_initial'].apply(lambda x: str(x))
    deepdf['PBSlocation'] = deepdf['PBSlocation'].apply(lambda x: str(x))
    deepdf['RT_initial_location'] = deepdf['RT_initial_location'].apply(lambda x: str(x))
    deepdf['RT_mutated_location'] = deepdf['RT_mutated_location'].apply(lambda x: str(x))
    deepdf['deepeditposition_lst'] = deepdf['deepeditposition_lst'].apply(lambda x: str(x))

    # set mt for deletions to 0:
    deepdf['edited_base_mt'] = deepdf.apply(lambda x: 0 if x.Correction_Type == 'Deletion' else x.edited_base_mt,
                                            axis=1)

    deepdf['original_base_mt'] = deepdf.apply(lambda x: 0 if x.Correction_Type == 'Insertion' else x.original_base_mt,
                                              axis=1)
    
    plain_tcols = ['averageedited', 'averageunedited', 'averageindel']
    for model_id, models_lst in models_lst_dict.items():
        cell_types=['K562','HEK']

        pred_dfs = [] # List to store prediction dataframes for each model

        for prieml_model, model_dir in models_lst: # Iterate over each model
            dloader = prieml_model.prepare_data(deepdf, 
                                                model_id,
                                                cell_types=cell_types, 
                                                y_ref=[], 
                                                batch_size=1500)
            
            # Predict using the current model
            pred_df = prieml_model.predict_from_dloader(dloader, model_dir, y_ref=plain_tcols)

            pred_df['model'] = model_id
            pred_dfs.append(pred_df) # Append the prediction dataframe to the list

        # split numeric and non-numeric data
        numeric_cols = ['pred_averageedited', 'pred_averageunedited', 'pred_averageindel']
        non_numeric_cols = ['seq_id', 'dataset_name', 'model']

        numeric_dfs = []
        for df in pred_dfs:
            numeric_dfs.append(df[numeric_cols])
        
        non_numeric_df = pred_dfs[0][non_numeric_cols]  # assuming non-numeric cols are the same in all dfs

        # convert list of DataFrames to a 3D numpy array
        arr = np.stack([df.values for df in numeric_dfs])

        # compute the mean over the first axis (the models)
        avg_preds_numeric = pd.DataFrame(arr.mean(axis=0), columns=numeric_cols)

        # combine averaged numeric predictions with non-numeric data
        avg_preds = pd.concat([non_numeric_df, avg_preds_numeric], axis=1)

        all_avg_preds[model_id] = avg_preds
    return all_avg_preds


def primerdesign(seq):
    '''Design NGS primers flanking the edit by at least +/- 20 bp'''
    try:
        print('Designing PCR primers...')
        seq_set = set(seq)
        if '/' in seq_set:
            original_bases = seq.split('/')[0].split('(')[1]
            if original_bases == '-':
                original_bases = ''
        elif '+' in seq_set:  #insertion
            original_bases = ''
        elif '-' in seq_set:  #deletion
            original_bases = seq.split('-')[1].split(')')[0]

        original_bases_length = len(original_bases)
        seq_before = seq.split('(')[0]
        seq_after = seq.split('(')[1].split(')')[1]
        original_seq = seq_before + original_bases + seq_after
        left_primer_boundary = len(seq_before) - 20
        right_primer_boundary = 20 + original_bases_length + 20

        seqdict = {'SEQUENCE_TEMPLATE': original_seq,
                   'SEQUENCE_TARGET': [left_primer_boundary, right_primer_boundary]}

        globargs = {
            'PRIMER_OPT_SIZE': 20,
            'PRIMER_PICK_INTERNAL_OLIGO': 0,
            'PRIMER_MIN_SIZE': 18,
            'PRIMER_MAX_SIZE': 25,
            'PRIMER_OPT_TM': 60.0,
            'PRIMER_MIN_TM': 56.0,
            'PRIMER_MAX_TM': 64.0,
            'PRIMER_MIN_GC': 20.0,
            'PRIMER_MAX_GC': 80.0,
            'PRIMER_MAX_POLY_X': 100,
            'PRIMER_SALT_MONOVALENT': 50.0,
            'PRIMER_DNA_CONC': 50.0,
            'PRIMER_MAX_NS_ACCEPTED': 0,
            'PRIMER_MAX_SELF_ANY': 12,
            'PRIMER_MAX_SELF_END': 8,
            'PRIMER_PAIR_MAX_COMPL_ANY': 12,
            'PRIMER_PAIR_MAX_COMPL_END': 8,
            'PRIMER_PRODUCT_SIZE_RANGE': [130, 200],
        }

        outcome = primer3.bindings.designPrimers(seqdict, globargs)
        primerdf = pd.DataFrame.from_dict(outcome, orient='index')

        primer_left0 = outcome['PRIMER_LEFT_0_SEQUENCE']
        primer_right0 = outcome['PRIMER_RIGHT_0_SEQUENCE']
        primer_pair_penalty0 = outcome['PRIMER_PAIR_0_PENALTY']
        primer_pair_length0 = outcome['PRIMER_PAIR_0_PRODUCT_SIZE']
        primerdf_short = pd.DataFrame()
        primerdf_short.loc['bestprimers', 'PRIMER_LEFT_0_SEQUENCE'] = primer_left0
        primerdf_short.loc['bestprimers', 'PRIMER_RIGHT_0_SEQUENCE'] = primer_right0
        primerdf_short.loc['bestprimers', 'PRIMER_PAIR_0_PENALTY'] = primer_pair_penalty0
        primerdf_short.loc['bestprimers', 'PRIMER_PAIR_0_PRODUCT_SIZE'] = int(primer_pair_length0)
    except:
        print('No PCR primers generated...')
        primerdf = pd.DataFrame()

        primer_left0 = ''
        primer_right0 = ''
        primer_pair_penalty0 = ''
        primer_pair_length0 = ''
        primerdf_short = pd.DataFrame()
        primerdf_short.loc['bestprimers', 'PRIMER_LEFT_0_SEQUENCE'] = primer_left0
        primerdf_short.loc['bestprimers', 'PRIMER_RIGHT_0_SEQUENCE'] = primer_right0
        primerdf_short.loc['bestprimers', 'PRIMER_PAIR_0_PENALTY'] = primer_pair_penalty0
        primerdf_short.loc['bestprimers', 'PRIMER_PAIR_0_PRODUCT_SIZE'] = ''

    return primerdf_short, primerdf


def parallel_batch_analysis(inp_dir, inp_fname, out_dir, out_fname, num_proc_arg, nicking, ngsprimer, run_ids=[0], combine_dfs=True):
    """Perform pegRNA predictions in batch-mode."""
    batchsequencedf = pd.read_csv(os.path.join(inp_dir, inp_fname))
    if 'editseq' in batchsequencedf:
        if 'sequence_name' in batchsequencedf:
            if len(batchsequencedf.sequence_name.unique()) == len(batchsequencedf.sequence_name):
                print(f'... Designing pegRNAs for {len(batchsequencedf)} sequences ...')
                try:
                    # make sequence_name column to string even if there are only numbers
                    batchsequencedf['sequence_name'] = batchsequencedf['sequence_name'].astype(str)
                    run_processing_parallel(batchsequencedf, out_dir, out_fname, num_proc_arg, nicking, ngsprimer, run_ids=run_ids, combine_dfs=combine_dfs)
                except ValueError:
                    print('***\n Error :( Check your input format is compatible with PRIDICT! More information in input box on https://pridict.it/ ...\n***\n')

            else:
                print('Please check your input-file! (Names not unique.')
        else:
            print('Please check your input-file! (Missing "sequence_name" column.')

    else:
        print('Please check your input-file! (Missing "editseq" column.')

def pegRNAfinder(dfrow, models_list, queue, pindx, pred_dir, nicking, ngsprimer,
                 editor='PE2-NGG', PBSlength_variants=PBSlengthrange, windowsize=windowsize_max,
                 RTseqoverhang_variants=RToverhanglengthrange):
    """Find pegRNAs and prediction scores for a set desired edit."""
    
    try:
        if type(dfrow['editseq']) == pd.Series: # case of dfrow is a group
            sequence = dfrow['editseq'].values[0]
            name = dfrow['sequence_name'].values[0]
        else: # case of dfrow is a row in a dataframe
            sequence = dfrow['editseq'] # string
            name = dfrow['sequence_name']

        start_time = time.time()
        original_base, edited_base, original_seq, edited_seq, editposition_left, editposition_right, mutation_type, correction_length, basebefore_temp, baseafter_temp = primesequenceparsing(
            sequence)
        if (editposition_left < 99) or (editposition_right< 99):
            print('Less than 100bp flanking sequence! Check your input.')
            raise ValueError
            
        sequence = sequence.upper()
        PAM, numberN, variant, protospacerlength, PAM_side, primescaffoldseq, PAM_length = editorcharacteristics(editor)

        mutationtypelist = []
        correctiontypelist = []
        correctionlengthlist = []
        edited_sequence_list = []  # original healthy sequence FW
        revcomp_edited_sequence_list = []  # original healthy sequence RV
        original_sequence_list = []  # mutated sequence FW
        revcomp_original_sequence_list = []  # mutated sequence RV
        mutation_position_to_PAM = []  # position of PAM motif respective to the mutation
        editedallelelist = []  # healthy base of the sequence
        originalallelelist = []  # mutated base of the sequence
        variantList = []  # Cas variant
        target_strandList = []  # defines which strand (FW or RV) should be targeted
        protospacerpamsequence = []
        protospacer_oligo_FW = []  # contains the protospacer oligo to be ordered for pegRNA cloning (FW)
        protospacer_oligo_RV = []  # contains the protospacer oligo to be ordered for pegRNA cloning (RV)
        extension_oligo_FW = []  # contains the extension oligo to be ordered for pegRNA cloning (FW)
        extension_oligo_RV = []  # contains the extension oligo to be ordered for pegRNA cloning (RV)
        editpositionlist = []
        PBSlength_variants_dic = {}
        PBSrevcomp_dic = {}
        for length in PBSlength_variants:
            PBSlength_variants_dic[length] = []
            PBSrevcomp_dic[length] = []
        RTseqoverhang_variants_dic = {}
        for length in RTseqoverhang_variants:
            RTseqoverhang_variants_dic[length] = []

        PBSsequencelist = []
        PBSrevcomplist = []
        RTseqlist = []
        RTseqoverhangrevcomplist = []
        RTseqrevcomplist = []
        deepcas9seqlist = []
        pbslengthlist = []
        rtlengthoverhanglist = []
        rtlengthlist = []
        pegRNA_list = []
        nickingprotospacerlist = []
        nickingdeepcas9list = []
        nickingpositiontoeditlist = []
        nickingtargetstrandlist = []
        mutation_position_to_nicklist = []
        nickingpe3blist = []
        nickingPAMdisruptlist = []
        nicking_oligo_FW = []
        nicking_oligo_RV = []
        protospacermtlist = []
        extensionmtlist = []
        RTmtlist = []
        RToverhangmtlist = []
        PBSmtlist = []
        original_base_mtlist = []
        edited_base_mtlist = []
        original_base_mt_nan_list = []
        edited_base_mt_nan_list = []
        rtoverhangmatchlist = []
        wide_initial_targetlist = []
        wide_mutated_targetlist = []
        protospacerlocation_only_initiallist = []
        PBSlocationlist = []
        RT_initial_locationlist = []
        RT_mutated_locationlist = []
        deepeditpositionlist = []
        multideepeditpositionlist = []

        if mutation_type == 'Deletion':
            correction_type = 'Deletion'
        elif mutation_type == 'Insertion':
            correction_type = 'Insertion'
        elif mutation_type == '1bpReplacement':
            correction_type = 'Replacement'
        elif mutation_type == 'MultibpReplacement':  # also set multibp replacements as replacements
            correction_type = 'Replacement'
        else:
            print('Editing type currently not supported for pegRNA prediction.')
            raise ValueError

        target_strandloop_start_time = time.time()
        for target_strand in ['Fw', 'Rv']:
            if target_strand == 'Fw':
                editposition = editposition_left
            else:
                editposition = editposition_right
                if not original_base == '-':
                    original_base = str(Seq(original_base).reverse_complement())
                if not edited_base == '-':
                    edited_base = str(Seq(edited_base).reverse_complement())
                original_seq = str(Seq(original_seq).reverse_complement())
                edited_seq = str(Seq(edited_seq).reverse_complement())

            editingWindow = range(0 - windowsize + 3, 4)  # determines how far the PAM can be away from the edit
            temp_dic = {}  # dictionary which contains many different key/value pairs which are later put together to the final lists

            X = [m.start() for m in re.finditer(PAM, original_seq)]
            X = [x for x in X if 25 <= x < len(
                original_seq) - 4]  # only use protospacer which have sufficient margin for 30bp sequence stretch
            
            # find PAMs in edited sequence for nicking guides
            editedstrand_PAMlist = [m.start() for m in re.finditer(PAM, edited_seq.upper())]
            editedstrand_PAMlist = [x for x in editedstrand_PAMlist if 25 <= x < len(edited_seq) - 4]  # only use protospacer which have sufficient margin for 30bp sequence stretch

            if X:
                xindex = 0
                editedstrandPAMindex = 0
                for editedstrandPAM in editedstrand_PAMlist:
                    editedPAM_int = editedstrand_PAMlist[editedstrandPAMindex] - editposition - numberN
                    editedstrandPAMindex = editedstrandPAMindex + 1
                    editedPAM = editedPAM_int + editposition + numberN
                    editedstart = editedPAM + (len(PAM) - 7) - 3
                    editednickposition = editedstart - editposition
                    editednickprotospacer, editednickdeepcas9 = nickingguide(edited_seq, editedstrandPAM, protospacerlength)
                    nickingtargetstrandlist.append(target_strand)
                    nickingpositiontoeditlist.append(editednickposition)
                    nickingprotospacerlist.append(editednickprotospacer)
                    nickingdeepcas9list.append(editednickdeepcas9)
                    if editednickprotospacer[0] != 'G':
                        editednickprotospacer = 'g'+editednickprotospacer
                    nicking_oligo_FW.append('cacc' + editednickprotospacer)
                    nicking_oligo_RV.append('aaac' + str(Seq(editednickprotospacer).reverse_complement()))
                    pe3bwindowlist = list(range(-5, 17))
                    pe3bwindowlist.remove(-3)

                    if editednickposition in pe3bwindowlist: # -5 or -4 for NGG or -2 to 14
                        nickingpe3blist.append('PE3b')
                        if editednickposition in [-5,-4]:
                            nickingPAMdisruptlist.append('Nicking_PAM_disrupt')
                        else:
                            nickingPAMdisruptlist.append('No_nicking_PAM_disrupt')
                    else:
                        nickingpe3blist.append('No_PE3b')
                        nickingPAMdisruptlist.append('No_nicking_PAM_disrupt')

                for xvalues in X:
                    X_int = X[xindex] - editposition - numberN
                    xindex = xindex + 1
                    XPAM = X_int + editposition + numberN
                    start = XPAM + (len(PAM) - 7) - 3
                    if X_int in editingWindow:
                        # start coordinates of RT correspond to nick position within protospacer (based on start of input sequence)
                        RTseq = {}
                        RTseqrevcomp = {}
                        RTseqoverhang = {}
                        RTseqoverhangrevcomp = {}
                        RTseqlength = {}

                        for RTlengthoverhang in RTseqoverhang_variants:  # loop which creates dictionaries containing RTseq and RTseqoverhang sequences for all different RT length specified in RTseqoverhang_variants
                            stop = editposition + len(edited_base) + RTlengthoverhang
                            if edited_base == '-':
                                stop -= 1

                            RTseq[RTlengthoverhang] = edited_seq[start:stop]
                            RTseqlength[RTlengthoverhang] = len(
                                RTseq[RTlengthoverhang])  # length of total RTseq (not only overhang)
                            RTseqoverhang[RTlengthoverhang] = edited_seq[stop - RTlengthoverhang:stop]
                            RToverhangstartposition = stop - RTlengthoverhang  # where RToverhang starts

                            RTseqrevcomp[RTlengthoverhang] = str(Seq(RTseq[RTlengthoverhang]).reverse_complement())
                            RTseqoverhangrevcomp[RTlengthoverhang] = str(
                                Seq(RTseqoverhang[RTlengthoverhang]).reverse_complement())

                        protospacerseq = 'G' + original_seq[XPAM + (len(PAM) - 7) - protospacerlength:XPAM + (len(PAM) - 7)] # attach G at position 1 to all protospacer
                        protospacerrev = Seq(protospacerseq).reverse_complement()
                        deepcas9seq = original_seq[XPAM + (len(PAM) - 12) - protospacerlength:XPAM + (len(PAM) - 1)]

                        # design different PBS lengths:
                        PBS = {}
                        PBSrevcomp = {}
                        pegRNA_dic = {}
                        # create pegRNA dictionary based on PBS/RT sequence dictionaries created above
                        for PBSlength in PBSlength_variants:
                            PBS[PBSlength] = original_seq[
                                            XPAM + (len(PAM) - 7) - protospacerlength + (
                                                    protospacerlength - PBSlength) - 3:XPAM + (len(PAM) - 7) - 3]
                            PBSrevcomp[PBSlength] = str(Seq(PBS[PBSlength]).reverse_complement())
                            
                            if 'PBS' + str(
                                    PBSlength) + 'revcomplist_temp' in temp_dic:  # only start appending after first round

                                temp_dic['PBS' + str(PBSlength) + 'revcomplist_temp'] = [
                                    temp_dic['PBS' + str(PBSlength) + 'revcomplist_temp'], PBSrevcomp[PBSlength]]
                                temp_dic['PBS' + str(PBSlength) + 'sequence_temp'] = [
                                    temp_dic['PBS' + str(PBSlength) + 'sequence_temp'], PBS[PBSlength]]
                            else:
                                temp_dic['PBS' + str(PBSlength) + 'revcomplist_temp'] = PBSrevcomp[PBSlength]
                                temp_dic['PBS' + str(PBSlength) + 'sequence_temp'] = PBS[PBSlength]
                            
                            for RTlengthoverhang in RTseqoverhang_variants:
                                pegRNA_dic['pegRNA_' + str(PBSlength) + str('_') + str(
                                    RTlengthoverhang)] = protospacerseq + primescaffoldseq + RTseqrevcomp[
                                    RTlengthoverhang] + PBSrevcomp[PBSlength]
                                        
                                ###### Adding DeepLearning features/sequences:
                                startposition = 10

                                wide_initial_target = original_seq[XPAM + (len(PAM) - 7) - protospacerlength - 10:]
                                wide_initial_target = wide_initial_target[:99]
                                wide_initial_targetlist.append(wide_initial_target)

                                wide_mutated_target = edited_seq[XPAM + (len(PAM) - 7) - protospacerlength - 10:]
                                wide_mutated_target = wide_mutated_target[:99]
                                wide_mutated_targetlist.append(wide_mutated_target)

                                deepeditposition = startposition + protospacerlength - 3 + ((X_int - 3) * -1)
                                deepeditpositionlist.append(deepeditposition)

                                if mutation_type == 'MultibpReplacement':
                                    multideepeditpositions = multideepeditpositionfunc(original_base, edited_base, deepeditposition)
                                    multideepeditpositionlist.append(multideepeditpositions)
                                else:
                                    multideepeditpositionlist.append([deepeditposition])

                                protospacerlocation_only_initial = [startposition, startposition + protospacerlength]
                                protospacerlocation_only_initiallist.append(protospacerlocation_only_initial)

                                PBSlocation = [startposition + protospacerlength - 3 - PBSlength,
                                            startposition + protospacerlength - 3]
                                PBSlocationlist.append(PBSlocation)

                                if correction_type == 'Replacement':
                                    RT_initial_location = [startposition + protospacerlength - 3,
                                                        startposition + protospacerlength - 3 + len(
                                                            RTseq[RTlengthoverhang])]
                                    RT_mutated_location = [startposition + protospacerlength - 3,
                                                        startposition + protospacerlength - 3 + len(
                                                            RTseq[RTlengthoverhang])]
                                elif correction_type == 'Deletion':
                                    RT_initial_location = [startposition + protospacerlength - 3,
                                                        startposition + protospacerlength - 3 + len(
                                                            RTseq[RTlengthoverhang]) + correction_length]
                                    RT_mutated_location = [startposition + protospacerlength - 3,
                                                        startposition + protospacerlength - 3 + len(
                                                            RTseq[RTlengthoverhang])]

                                elif correction_type == 'Insertion':
                                    RT_initial_location = [startposition + protospacerlength - 3,
                                                        startposition + protospacerlength - 3 + len(
                                                            RTseq[RTlengthoverhang]) - correction_length]
                                    RT_mutated_location = [startposition + protospacerlength - 3,
                                                        startposition + protospacerlength - 3 + len(
                                                            RTseq[RTlengthoverhang])]

                                RT_initial_locationlist.append(RT_initial_location)
                                RT_mutated_locationlist.append(RT_mutated_location)

                                ######

                                extension_oligo_FW.append(
                                    primescaffoldseq[-4:] + RTseqrevcomp[RTlengthoverhang] + PBSrevcomp[PBSlength])
                                extension_oligo_RV.append('AAAA' + str(PBS[PBSlength]) + str(RTseq[RTlengthoverhang]))
                                RTseqrevcomplist.append(RTseqrevcomp[RTlengthoverhang])
                                RTseqlist.append(RTseq[RTlengthoverhang])
                                RTseqoverhangrevcomplist.append(RTseqoverhangrevcomp[RTlengthoverhang])
                                pbslengthlist.append(
                                    PBSlength)  # add PBS length to list and to table in the end
                                rtlengthoverhanglist.append(
                                    RTlengthoverhang)  # add RT overhang length to list and to table in the end
                                rtlengthlist.append(RTseqlength[RTlengthoverhang])
                                target_strandList.append(target_strand)
                                protospacerpamsequence.append(protospacerseq)
                                protospacer_oligo_FW.append('CACC' + protospacerseq + primescaffoldseq[:5])
                                protospacer_oligo_RV.append(
                                    str(Seq(primescaffoldseq[:9]).reverse_complement()) + str(protospacerrev))
                                deepcas9seqlist.append(deepcas9seq)
                                PBSsequencelist.append(PBS[PBSlength])
                                PBSrevcomplist.append(PBSrevcomp[PBSlength])
                                editpositionlist.append(editposition - start)
                                pegRNA = protospacerseq + primescaffoldseq + RTseqrevcomp[RTlengthoverhang] + PBSrevcomp[
                                    PBSlength]
                                pegRNA_list.append(pegRNA)
                                edited_sequence_list.append(edited_seq)
                                revcomp_edited_sequence_list.append(str(Seq(edited_seq).reverse_complement()))
                                original_sequence_list.append(original_seq)
                                revcomp_original_sequence_list.append(str(Seq(original_seq).reverse_complement()))
                                variantList.append(variant)
                                editedallelelist.append(edited_base)
                                originalallelelist.append(original_base)
                                mutationtypelist.append(mutation_type)
                                correctiontypelist.append(correction_type)
                                mutation_position_to_PAM.append(X_int)
                                mutation_position_to_nicklist.append(X_int - 3)
                                correctionlengthlist.append(correction_length)
                                protospacermt, extensionmt, RTmt, RToverhangmt, PBSmt, original_base_mt, edited_base_mt, original_base_mt_nan, edited_base_mt_nan = melting_temperature(
                                    protospacerseq, RTseqrevcomp[RTlengthoverhang] + PBSrevcomp[PBSlength],
                                    RTseqrevcomp[RTlengthoverhang], RTseqoverhangrevcomp[RTlengthoverhang],
                                    PBSrevcomp[PBSlength], original_base, edited_base)
                                protospacermtlist.append(protospacermt)
                                extensionmtlist.append(extensionmt)
                                RTmtlist.append(RTmt)
                                RToverhangmtlist.append(RToverhangmt)
                                PBSmtlist.append(PBSmt)
                                original_base_mtlist.append(original_base_mt)
                                edited_base_mtlist.append(edited_base_mt)
                                original_base_mt_nan_list.append(original_base_mt_nan)
                                edited_base_mt_nan_list.append(edited_base_mt_nan)

                                rtoverhangmatch = RToverhangmatches(RTseqoverhang[RTlengthoverhang], edited_seq,
                                                                    RToverhangstartposition, RTlengthoverhang)
                                rtoverhangmatchlist.append(rtoverhangmatch)


        target_strandlooptime = time.time() - target_strandloop_start_time   
        start_time = time.time()
        if nicking:
            nickingdeepcas9scorelist = deepcas9(nickingdeepcas9list)
        start_time = time.time()
        if ngsprimer:
            primerdf_short, primerdf = primerdesign(sequence)  # design PCR primers for targetseq

        ### check whether deepcas9 sequence was made the same for the training set as here!

        if nicking:
            nickingdf = pd.DataFrame(
                {'Nicking-Protospacer': nickingprotospacerlist, 'Nicking-Position-to-edit': nickingpositiontoeditlist,
                'PE3b': nickingpe3blist,
                'Nicking-PAMdisrupt': nickingPAMdisruptlist, 'Target_Strand': nickingtargetstrandlist,
                '30bpseq': nickingdeepcas9list, 'DeepCas9score': nickingdeepcas9scorelist,
                'Nicking-Proto-Oligo-FW': nicking_oligo_FW,
                'Nicking-Proto-Oligo-RV': nicking_oligo_RV})

        pegdataframe = pd.DataFrame({'Original_Sequence': original_sequence_list, 'Edited-Sequences': edited_sequence_list,
                                    'Target-Strand': target_strandList, 'Mutation_Type': mutationtypelist,
                                    'Correction_Type': correctiontypelist, 'Correction_Length': correctionlengthlist,
                                    'Editing_Position': editpositionlist,
                                    'PBSlength': pbslengthlist, 'RToverhanglength': rtlengthoverhanglist,
                                    'RTlength': rtlengthlist, 'EditedAllele': editedallelelist,
                                    'OriginalAllele': originalallelelist, 'Spacer-Sequence': protospacerpamsequence,
                                    'PBSrevcomp': PBSrevcomplist, 'RTseqoverhangrevcomp': RTseqoverhangrevcomplist,
                                    'RTrevcomp': RTseqrevcomplist,
                                    'Spacer-Oligo-FW': protospacer_oligo_FW,
                                    'Spacer-Oligo-RV': protospacer_oligo_RV,
                                    'Extension-Oligo-FW': extension_oligo_FW, 'Extension-Oligo-RV': extension_oligo_RV,
                                    'pegRNA': pegRNA_list,
                                    'Editor_Variant': variantList,
                                    'protospacermt': protospacermtlist,
                                    'extensionmt': extensionmtlist, 'RTmt': RTmtlist, 'RToverhangmt': RToverhangmtlist,
                                    'PBSmt': PBSmtlist,
                                    'original_base_mt': original_base_mtlist, 'edited_base_mt': edited_base_mtlist,
                                    'original_base_mt_nan': original_base_mt_nan_list, 'edited_base_mt_nan': edited_base_mt_nan_list,
                                    'RToverhangmatches': rtoverhangmatchlist,
                                    'wide_initial_target': wide_initial_targetlist,
                                    'wide_mutated_target': wide_mutated_targetlist,
                                    'protospacerlocation_only_initial': protospacerlocation_only_initiallist,
                                    'PBSlocation': PBSlocationlist, 'RT_initial_location': RT_initial_locationlist,
                                    'RT_mutated_location': RT_mutated_locationlist,
                                    'deepeditposition': deepeditpositionlist, 'deepeditposition_lst': multideepeditpositionlist})
        

        if len(pegdataframe) < 1:
            print('\n***\nNo PAM (NGG) found in proximity of edit!\n***\n')
            raise ValueError
                
        start_time = time.time()

        
        all_avg_preds = deeppridict(pegdataframe, models_list)


        # Extracting cell types from model
        cell_types=['K562','HEK']

        # Inserting common columns outside the loop
        pegdataframe.insert(len(pegdataframe.columns), 'sequence_name', name)

        average_editing_scores = {cell_type: [] for cell_type in cell_types}
        # average_unintended_scores = {cell_type: [] for cell_type in cell_types}

        for model_id, pred_df in all_avg_preds.items():
            # Looping over each cell type
            for cell_type in cell_types:
                # Filtering the dataframe based on cell type
                pred_df_cell = pred_df[pred_df['dataset_name'] == cell_type]
                
                # Calculating the editing prediction lists
                editingpredictionlist = (pred_df_cell['pred_averageedited']*100).tolist()
                # unintendededitingpredictionlist = (pred_df_cell['pred_averageindel']*100).tolist()

                average_editing_scores[cell_type].append(editingpredictionlist)
                # average_unintended_scores[cell_type].append(unintendededitingpredictionlist)

                # Inserting cell-type specific columns into pegdataframe
                pegdataframe.insert(len(pegdataframe.columns), f'{model_id}_editing_Score_deep_{cell_type}', editingpredictionlist)
                # pegdataframe.insert(len(pegdataframe.columns), f'{model_id}_unintended_Score_deep_{cell_type}', unintendededitingpredictionlist)

        for cell_type in cell_types:
            average_editing = np.mean(average_editing_scores[cell_type], axis=0).tolist()
            # average_unintended = np.mean(average_unintended_scores[cell_type], axis=0).tolist()

            pegdataframe.insert(len(pegdataframe.columns), f'PRIDICT2_0_editing_Score_deep_{cell_type}', average_editing)
            # pegdataframe.insert(len(pegdataframe.columns), f'PRIDICT2_0_unintended_Score_deep_{cell_type}', average_unintended)

        def find_closest_percentile(value, ref_column, percentile_column):
            closest_index = np.abs(ref_column - value).idxmin()
            return percentile_column.loc[closest_index]  
        libdiversedf = pd.read_csv('dataset/20230913_Library_Diverse_Ranking_Percentile.csv')    
        for cell_type in cell_types:
            pegdataframe.sort_values(by=[f'PRIDICT2_0_editing_Score_deep_{cell_type}'], inplace=True, ascending=False)
            pegdataframe[f'{cell_type}_rank'] = range(1, len(pegdataframe) + 1)
            pegdataframe[f'{cell_type}_percentile_to_librarydiverse'] = pegdataframe[f'PRIDICT2_0_editing_Score_deep_{cell_type}'].apply(
                    lambda x: find_closest_percentile(
                        x, 
                        libdiversedf[f'{cell_type}averageedited'], 
                        libdiversedf[f'{cell_type}percentile']
                    )
                )

       
        # The time calculation and print statements
        pridict_time = time.time() - start_time
        print()
        print("Calculating features took", round(target_strandlooptime,3), "seconds to run.")
        print("Deep model took", round(pridict_time,1), "seconds to run.")

        if nicking:
            nickingdf.sort_values(by=['DeepCas9score'], inplace=True, ascending=False)
            nickingdf.to_csv(os.path.join(pred_dir, name + '_nicking_guides.csv'))
        

        pegdataframe.to_csv(os.path.join(pred_dir, name + '_pegRNA_Pridict_full.csv'))
        if ngsprimer:
            primerdf_short.to_csv(os.path.join(pred_dir, name + '_best_PCR_primers.csv'))

    except Exception as e:
        print('-- Exception occured --')
        print(e)
    finally:
        queue.put(pindx)

# editseq_test = 'GCCTGGAGGTGTCTGGGTCCCTCCCCCACCCGACTACTTCACTCTCTGTCCTCTCTGCCCAGGAGCCCAGGATGTGCGAGTTCAAGTGCTACCCGA(G/C)GTGCGAGGCCAGCTCGGGGGCACCGTGGAGCTGCCGTGCCACCTGCTGCCACCTGTTCCTGGACTGTACATCTCCCTGGTGACCTGGCAGCGCCCAGATGCACCTGCGAACCACCAGAATGTGGCCGC'

def fix_mkl_issue():
    # https://github.com/pytorch/pytorch/issues/37377
    # https://github.com/IntelPython/mkl-service/issues/14
    import mkl
    a, b, __ = mkl.__version__.split('.')
    if int(a) <=2 and int(b)<=3:
        print('setting MKL_THREADING_LAYER = GNU')
        os.environ['MKL_THREADING_LAYER'] = 'GNU'

def run_processing_parallel(df, pred_dir, fname, num_proc_arg, nicking, ngsprimer, run_ids=[0], combine_dfs=True):

    fix_mkl_issue() # comment this 
    queue = mp.Queue()
    q_processes = []
    if num_proc_arg == 0:
        num_proc = mp.cpu_count()
    elif num_proc_arg <= mp.cpu_count():
        num_proc = num_proc_arg
    else:
        num_proc = mp.cpu_count()


    num_rows = len(df)
    seqnames_lst = []
    models_lst = load_pridict_model(run_ids = run_ids)

    for q_i in range(min(num_proc, num_rows)):
        # print('q_i:', q_i)
        row = df.iloc[q_i] # slice a row
        seqnames_lst.append(row['sequence_name'])
        print('processing sequence:', seqnames_lst[-1])
        q_process = create_q_process(dfrow=row,
                                     models_list=models_lst,
                                     queue=queue,
                                     pindx=q_i,
                                     pred_dir=pred_dir, nicking=nicking, ngsprimer=ngsprimer)
        q_processes.append(q_process)
        spawn_q_process(q_process)

    spawned_processes = min(num_proc, num_rows)

    print("*"*25)
    for q_i in range(num_rows):
        join_q_process(q_processes[q_i])
        released_proc_num = queue.get()
        # print("released_process_num:", released_proc_num)
        q_processes[q_i] = None # free resources ;)
        if(spawned_processes < num_rows):
            q_i_upd = q_i + num_proc
            # print('q_i:', q_i, 'q_i_updated:', q_i_upd)
            row = df.iloc[q_i_upd]
            seqnames_lst.append(row['sequence_name'])
            print('processing sequence:', seqnames_lst[-1])
            q_process = create_q_process(dfrow=row, 
                                         models_list=models_lst,
                                         queue=queue,
                                         pindx=q_i_upd,
                                         pred_dir=pred_dir, nicking=nicking, ngsprimer=ngsprimer)

            q_processes.append(q_process)
            spawn_q_process(q_process)
            spawned_processes = spawned_processes + 1
    
    # assemble all sequence dataframes into one --- optional
    if combine_dfs:
        combined_df = assemble_df(seqnames_lst, 'pegRNA_Pridict_full', pred_dir)
        remove_col(combined_df, 'Unnamed: 0')
        
        combined_df.to_csv(os.path.join(pred_dir, f'{fname}_pegRNA_Pridict_full.csv'), index=False)

        cols = get_shortdf_colnames()
        pegdataframe_short = combined_df[cols]
        pegdataframe_short.to_csv(os.path.join(pred_dir, f'{fname}_pegRNA_Pridict_Scores.csv'), index=False)

        combined_bestpegdf = assemble_df(seqnames_lst, 'best_pegdf', pred_dir)
        combined_bestpegdf.to_csv(os.path.join(pred_dir, f'{fname}_best_pegdf.csv'), index=False)


def remove_col(df, colname):
    if colname in df:
        del df[colname]

def get_shortdf_colnames():
    models = ['PRIDICT1_1', 'PRIDICT1_2', 'PRIDICT2_0']  # List of models you are using
    cell_types = ['K562', 'HEK']
    base_cols = [
        'Protospacer-Oligo-FW', 'Protospacer-Oligo-RV', 'Extension-Oligo-FW',
        'Extension-Oligo-RV', 'Original_Sequence', 'Edited-Sequences',
        'Target-Strand', 'Mutation_Type', 'Correction_Type', 'Correction_Length',
        'Editing_Position', 'PBSlength', 'RToverhanglength', 'RTlength',
        'EditedAllele', 'OriginalAllele', 'Spacer-Sequence',
        'PBSrevcomp', 'RTseqoverhangrevcomp', 'RTrevcomp',
        'pegRNA', 'Editor_Variant', 'sequence_name'
    ]
    
    # Add PRIDICT scores for each model and cell type
    pridict_cols = []
    for model in models:
        for cell_type in cell_types:
            pridict_cols.append(f'{model}_editing_Score_deep_{cell_type}')
            # pridict_cols.append(f'{model}_unintended_Score_deep_{cell_type}')
    
    # Add rank columns for each cell type
    rank_cols = [f'{cell_type}_rank' for cell_type in cell_types]
    
    return base_cols + pridict_cols + rank_cols

def assemble_df(seqnames_lst, namesuffix, df_dir):
    df_lst = []
    for seq_name in seqnames_lst:
        fpath = os.path.join(df_dir, f'{seq_name}_{namesuffix}.csv')
        if os.path.isfile(fpath):
            df =  pd.read_csv(fpath)
            df['sequence_name'] = seq_name
            df_lst.append(df)
    combined_df = pd.concat(df_lst, axis=0)
    return combined_df


def spawn_q_process(q_process):
    print(">>> spawning row computation process")
    q_process.start()
    
def join_q_process(q_process):
    q_process.join()
    print("<<< joined row computation process")
    
def create_q_process(dfrow, models_list, queue, pindx, pred_dir, nicking, ngsprimer):
    return mp.Process(target=pegRNAfinder, args=(dfrow, models_list, queue, pindx, pred_dir, nicking, ngsprimer))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running PRIDICT to design and predict pegRNAs.")

    subparser = parser.add_subparsers(dest='command')
    manual_m = subparser.add_parser('manual')
    batch_m  = subparser.add_parser('batch')

    manual_m.add_argument("--sequence-name", type=str, help="Name of the sequence (i.e. unique id for the sequence)", required=True)
    manual_m.add_argument("--sequence", type=str, help="Target sequence to edit (format: xxxxxxxxx(a/g)xxxxxxxxxx). Use quotation marks before and after the sequence.", required=True)
    manual_m.add_argument("--use_5folds", action='store_true', help="Use all 5-folds trained models (and average output). Default is to use fold-1 model.")
    manual_m.add_argument("--nicking", action='store_true', help="Additionally, design nicking guides for edit (PE3) with DeepSpCas9 prediction.")
    manual_m.add_argument("--ngsprimer", action='store_true', help="Additionally, design NGS primers for edit based on Primer3 design.")


    batch_m.add_argument("--input-fname", type=str, required=True, help="Input filename - name of csv file that has two columns {editseq, sequence_name}. See batch_template.csv in the ./input folder ")
    batch_m.add_argument("--output-fname", type=str, help="Output filename for the resulting dataframe. If not specified, the name of the input file will be used")
    batch_m.add_argument("--use_5folds", action='store_true', help="Use all 5-folds trained models (and average output). Default is to use fold-1 model")
    batch_m.add_argument("--combine_results", action='store_true', help="Compile all results into one dataframe")
    batch_m.add_argument("--nicking", action='store_true', help="Additionally, design nicking guides for edit (PE3) with DeepSpCas9 prediction.")
    batch_m.add_argument("--ngsprimer", action='store_true', help="Additionally, design NGS primers for edit based on Primer3 design.")
    batch_m.add_argument("--cores", type=int, default=0, help="Number of cores to use for multiprocessing. Default value uses 3 available cores. Maximum 3 cores to prevent memory issues.")
    

    args = parser.parse_args()

    if args.command == 'manual':
        print('Running in manual mode:')
        df = pd.DataFrame({'sequence_name': [args.sequence_name], 'editseq': [args.sequence]})

        out_dir = os.path.join(os.path.dirname(__file__), './predictions')
        print('output directory:', out_dir)

        fname = f'{args.sequence_name}'
        if args.use_5folds:
            run_ids = list(range(5))
        else:
            run_ids = [0]
        
        num_proc_arg = 1
        
        if args.nicking:
            nicking=True
        else:
            nicking=False
            
        if args.ngsprimer:
            ngsprimer=True
        else:
            ngsprimer=False
        
        run_processing_parallel(df, out_dir, fname, num_proc_arg, nicking, ngsprimer, run_ids=run_ids, combine_dfs=False)

    elif args.command == 'batch':
        print('Running in batch mode:')

        inp_dir = os.path.join(os.path.dirname(__file__), './input')
        print('input directory:', inp_dir)

        inp_fname = args.input_fname

        out_dir = os.path.join(os.path.dirname(__file__), './predictions')
        print('output directory:', out_dir)

        if args.use_5folds:
            run_ids = list(range(5))
        else:
            run_ids = [0]

        if args.output_fname:
            out_fname = args.output_fname
        else:
            out_fname = args.input_fname.split('.')[0]
        
        # max 3 cores to prevent memory issues
        if args.cores:
            num_proc_arg=args.cores
            if num_proc_arg > 3:
                num_proc_arg=3
        else:
            num_proc_arg=3
            
        if args.nicking:
            nicking=True
        else:
            nicking=False
            
        if args.ngsprimer:
            ngsprimer=True
        else:
            ngsprimer=False

        parallel_batch_analysis(inp_dir, inp_fname, out_dir, out_fname, num_proc_arg, nicking, ngsprimer, run_ids=run_ids, combine_dfs=args.combine_results)
    else:
        print('Please specify how to run PRIDICT2.0 ("manual" or "batch") as argument after the script name.')