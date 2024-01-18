### run the following python script on the cloud in the folder or PRIDICT2.0

from Bio.Seq import Seq
import pandas as pd
import os
import torch.multiprocessing as mp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mp.set_start_method("spawn", force=True)
import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", "Function deprecated please use \"design_primers\" instead", category=UserWarning)
from pridict.pridictv2.utilities import *
from pridict.pridictv2.dataset import *
from pridict.pridictv2.predict_outcomedistrib import *


def get_prieml_model_template():
    device = get_device(True, 0)
    wsize = 20
    normalize_opt = 'max'
    # create a model template that will be used to load/instantiate a target trained model
    prieml_model = PRIEML_Model(device, wsize=wsize, normalize=normalize_opt, fdtype=torch.float32)
    return prieml_model

def load_pridict_model(run_ids=[0], model_type='base_90k'):
    """construct and return PRIDICT model along with model files directory """
    models_lst_dict = {}  # Initialize a dictionary to hold lists of models keyed by model_id
    repo_dir = os.path.join(os.path.abspath('./'))
    if model_type == 'base_90k':
        modellist = [
        ('base_90k', 'pe_rnn_kldiv', 'exp_2023-06-02_09-49-21')
        ]
    elif model_type == 'base_390k':
        modellist = [
            ('base_390k', 'pe_rnn_distribution_multidata', 'xp_2023-08-26_20-58-14')
        ]
    
    # create a model template that will be used to load/instantiate a target trained model
    prieml_model = get_prieml_model_template()

    for model_desc_tup in modellist:
        models_lst = []  # Initialize models_lst for each model
        model_id, __, mfolder = model_desc_tup

        for run_num in run_ids: # add the different model runs (i.e. based on 5 folds)
            print('model_id:',model_id)
            print('mfolder:',mfolder)
            print('run_num:',run_num)
            model_dir = os.path.join(repo_dir, 'trained_models', model_id.lower(), mfolder, 'train_val', f'run_{run_num}')
            print(model_dir)
            cell_types = get_cell_types(model_type)
            loaded_model = prieml_model.build_retrieve_models(model_dir, cell_types)
            models_lst.append((loaded_model, model_dir))
        
        models_lst_dict[model_id] = models_lst  # Add to the dictionary

    return models_lst_dict

def deeppridict(pegdataframe, models_lst_dict, model_type='base_90k'):
    """Perform score prediction on dataframe of features based on RNN model.
    
    Args:
        pegdataframe: pandas DataFrame containing the processed sequence features
        models_lst: list of tuples of (pridict_model, model_run_dir)
    
    """

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
    deepdf.insert(1, 'seq_id', list(range(len(deepdf))))
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
    print(model_type)
    cell_types = get_cell_types(model_type)
    print(cell_types)
    batch_size = int(1500/len(cell_types))
    print('successful check -1')
    prieml_model = get_prieml_model_template()
    print('successful check 0')
    # print('deepdf[seq_id]:\n', deepdf['seq_id'])
    # data processing for the same data can be done once given that we already specified the cell_types a priori
    dloader = prieml_model.prepare_data(deepdf, 
                                        None, # since we are specifying cell types model_name can be ignored
                                        cell_types=cell_types, 
                                        y_ref=[], 
                                        batch_size=batch_size)
    
    print('successful check 1')
    all_avg_preds = {} 

    for model_id, model_runs_lst in models_lst_dict.items():
    
        pred_dfs = [] # List to store prediction dataframes for each model
        
        runs_c = 0
        print('successful check 2')
        for loaded_model_lst, model_dir in model_runs_lst: # Iterate over each model
            # Predict using the current model
            print(dloader)
            print('loaded_model_lst:',loaded_model_lst)
            pred_df = prieml_model.predict_from_dloader_using_loaded_models(dloader, loaded_model_lst, y_ref=plain_tcols)
            print('successful check 3')
            pred_df['run_num'] = runs_c # this is irrelevant as we will average at the end
            print('successful check 4')
            pred_dfs.append(pred_df) # Append the prediction dataframe to the list
            print('successful check 5')
            runs_c += 1
            # print('pred_df:\n', pred_df)
        # compuate average prediction across runs
        pred_df_allruns = pd.concat(pred_dfs, axis=0, ignore_index=True)
        avg_preds = prieml_model.compute_avg_predictions(pred_df_allruns)
        print('successful check 6')
        avg_preds['model'] = model_id
        print('successful check 7')
        # store the average prediction dataframe in for a specified model in a dictionary
        all_avg_preds[model_id] = avg_preds
        print('successful check 8')
    # print('all_avg_predicitons:\n', all_avg_preds)
    return all_avg_preds

def compute_average_predictions(df, grp_cols=['seq_id', 'dataset_name']):
    tcols = ['pred_averageedited', 'pred_averageunedited', 'pred_averageindel']
    agg_df = df.groupby(by=grp_cols)[tcols].mean()
    agg_df.reset_index(inplace=True)
    for colname in ('run_num', 'Unnamed: 0', 'model'):
        if colname in agg_df:
            del agg_df[colname]
    return agg_df

def get_cell_types(model_type):
    if model_type == 'base_90k':
        return ['HEK']
    elif model_type == 'base_390k':
        return ['HEKschwank','HEKhyongbum']

if __name__ == "__main__":
    for model_type in ['base_90k','base_390k']:
        # do 5-fold predictions
        run_ids=[0,1,2,3,4]
        models_lst_dict = load_pridict_model(run_ids = run_ids, model_type = model_type)
        pridict2_premadedf = pd.read_csv('input/20240113_librarydiv_df_batchfile_with_adapted_wide_initial_target.csv')
        all_avg_preds = deeppridict(pridict2_premadedf, models_lst_dict, model_type)

        # Extracting cell types from model
        cell_types = get_cell_types(model_type)

        tmp = [all_avg_preds[model_id] for model_id in all_avg_preds]
        # seq_id, dataset_name, model, predictions cols
        tmp_df = pd.concat(tmp, axis=0, ignore_index=True)
        agg_df = compute_average_predictions(tmp_df, grp_cols=['seq_id', 'dataset_name'])
        # print('agg_df:\n', agg_df)
        for cell_type in cell_types:
            cond  = agg_df['dataset_name'] == cell_type
            avg_edited_eff = agg_df.loc[cond, 'pred_averageedited'].values*100
            pridict2_premadedf.insert(len(pridict2_premadedf.columns), f'{model_type}_editing_Score_deep_{cell_type}', avg_edited_eff)
        pridict2_premadedf.to_csv(f'predictions/20240117_libdiverse_real_long_wide_target_with_{model_type}_model_predictions_5foldaverage.csv', index=False)
    
