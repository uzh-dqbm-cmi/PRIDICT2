import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

from .hyperparam import get_saved_config
from .utilities import ReaderWriter, build_predictions_df, check_na,switch_layer_to_traineval_mode,require_nonleaf_grad
from .data_preprocess import PESeqProcessor
from .dataset import create_datatensor, MinMaxNormalizer, create_datatensor_knockoff
from .rnn.rnn import RNN_Net

from .model import AnnotEmbeder_InitSeq, AnnotEmbeder_MutSeq, FeatureEmbAttention, \
                   MLPEmbedder, MLPDecoder, MaskGenerator
from .hyperparam import RNNHyperparamConfig
from .data_preprocess import Viz_PESeqs
from .feature_importance.baseline_generation import get_alpha_expanded, expand_tensor_msteps

class PRIEML_Model:
    def __init__(self, device, wsize=20, normalize='none', fdtype=torch.float32):
        self.device = device
        self.wsize = wsize
        self.normalize = normalize
        self.fdtype = fdtype

    def _process_df(self, df):
        """
        Args:
            df: pandas dataframe
        """
        print('--- processing input data frame ---')
        normalize = self.normalize
        assert normalize in {'none', 'max', 'minmax'}

        pe_seq_processor=PESeqProcessor()
        df = df.copy()
        # reset index in order to perform proper operations down the stream !
        df.reset_index(inplace=True, drop=True)
        if 'correction_type_categ' not in df:
            print('--- creating correction type categories ---')
            correction_categs = ['Deletion', 
                                'Insertion', 
                                'Replacement']
            df['correction_type_categ'] = pd.Categorical(df['Correction_Type'], categories=correction_categs)
            correction_type_df = pd.get_dummies(df['correction_type_categ'], prefix='Correction', prefix_sep='_')
            df = pd.concat([df, correction_type_df], axis=1)
        if 'seq_id' not in df:
            print('--- creating seq_id ---')
            df['seq_id'] = [f'seq_{i}' for i in range(df.shape[0])]

        # retrieve continuous column names
        minmax_normalizer = MinMaxNormalizer()
        norm_colnames = minmax_normalizer.get_colnames()

        proc_seq_init_df,num_init_cols, proc_seq_mut_df, num_mut_cols = pe_seq_processor.process_init_mut_seqs(df,
                                                                                                               ['seq_id'], 
                                                                                                               'wide_initial_target',
                                                                                                               'wide_mutated_target')
        # add PBSlength in case it is part of colnames
        if 'PBSlength' in norm_colnames and 'PBSlength' not in df:
            print('--- creating PBSlength ---')
            df['PBSlength'] = proc_seq_init_df['end_PBS'] - proc_seq_init_df['start_PBS']
        if normalize != 'none':
            print('--- normalizing continuous features ---')
            norm_colnames = minmax_normalizer.normalize_cont_cols(df, normalize_opt=normalize, suffix='_norm')
        # make sure everything checks out
        check_na(proc_seq_init_df)
        check_na(proc_seq_mut_df)
        check_na(df)
        return norm_colnames, df, proc_seq_init_df,num_init_cols, proc_seq_mut_df, num_mut_cols

    def _construct_datatensor(self, norm_colnames, df, proc_seq_init_df,num_init_cols, proc_seq_mut_df, num_mut_cols, maskpos_indices=None,y_ref=[]):
        print('--- creating datatensor ---')
        wsize=self.wsize # to read this from options dumped on disk
        if maskpos_indices is None:
            dtensor = create_datatensor(df, 
                                        proc_seq_init_df, num_init_cols, 
                                        proc_seq_mut_df, num_mut_cols,
                                        norm_colnames,
                                        window=wsize, 
                                        y_ref=y_ref)
        else:
            dtensor = create_datatensor_knockoff(df, 
                                        proc_seq_init_df, num_init_cols, 
                                        proc_seq_mut_df, num_mut_cols,
                                        norm_colnames,
                                        maskpos_indices,
                                        window=wsize,
                                        y_ref=y_ref)
        return dtensor

    def _construct_dloader(self, dtensor, batch_size):
        print('--- creating datatloader ---')
        dloader = DataLoader(dtensor,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            sampler=None)
        return dloader

    def _load_model_config(self, mconfig_dir):
        print('--- loading model config ---')
        mconfig, options = get_saved_config(mconfig_dir)
        return mconfig, options

    def _build_base_model(self, config):
        print('--- building model ---')
        device = self.device
        mconfig, options = config
        model_config = mconfig['model_config']
        
        # fdtype = options.get('fdtype')
        fdtype = self.fdtype
        annot_embed = options.get('annot_embed')
        assemb_opt = options.get('assemb_opt')
        seqlevel_featdim = options.get('seqlevel_featdim')

        init_annot_embed = AnnotEmbeder_InitSeq(embed_dim=model_config.embed_dim,
                                                annot_embed=annot_embed,
                                                assemb_opt=assemb_opt)
        mut_annot_embed = AnnotEmbeder_MutSeq(embed_dim=model_config.embed_dim,
                                              annot_embed=annot_embed,
                                              assemb_opt=assemb_opt)
        if assemb_opt == 'stack':
            init_embed_dim = model_config.embed_dim + 3*annot_embed
            mut_embed_dim = model_config.embed_dim + 2*annot_embed
            z_dim = np.min([init_embed_dim, mut_embed_dim])//2
        else:
            init_embed_dim = model_config.embed_dim
            mut_embed_dim = model_config.embed_dim
            z_dim = np.min([init_embed_dim, mut_embed_dim])//2 

        init_encoder = RNN_Net(input_dim =init_embed_dim,
                              hidden_dim=model_config.embed_dim,
                              z_dim=z_dim,
                              device=device,
                              num_hiddenlayers=model_config.num_hidden_layers,
                              bidirection=model_config.bidirection,
                              rnn_pdropout=model_config.p_dropout,
                              rnn_class=model_config.rnn_class,
                              nonlinear_func=model_config.nonlin_func,
                              fdtype=fdtype)
        mut_encoder= RNN_Net(input_dim =mut_embed_dim,
                              hidden_dim=model_config.embed_dim,
                              z_dim=z_dim,
                              device=device,
                              num_hiddenlayers=model_config.num_hidden_layers,
                              bidirection=model_config.bidirection,
                              rnn_pdropout=model_config.p_dropout,
                              rnn_class=model_config.rnn_class,
                              nonlinear_func=model_config.nonlin_func,
                              fdtype=fdtype)

        local_featemb_init_attn = FeatureEmbAttention(z_dim)
        local_featemb_mut_attn = FeatureEmbAttention(z_dim)

        global_featemb_init_attn = FeatureEmbAttention(z_dim)
        global_featemb_mut_attn = FeatureEmbAttention(z_dim)

        seqlevel_featembeder = MLPEmbedder(inp_dim=seqlevel_featdim,
                                           embed_dim=z_dim,
                                           mlp_embed_factor=1,
                                           nonlin_func=model_config.nonlin_func,
                                           pdropout=model_config.p_dropout, 
                                           num_encoder_units=1)

        decoder  = MLPDecoder(5*z_dim,
                              embed_dim=z_dim,
                              outp_dim=1,
                              mlp_embed_factor=2,
                              nonlin_func=model_config.nonlin_func, 
                              pdropout=model_config.p_dropout, 
                              num_encoder_units=1)

        # define optimizer and group parameters
        models = ((init_annot_embed, 'init_annot_embed'), 
                  (mut_annot_embed, 'mut_annot_embed'),
                  (init_encoder, 'init_encoder'),
                  (mut_encoder, 'mut_encoder'),
                  (local_featemb_init_attn, 'local_featemb_init_attn'),
                  (local_featemb_mut_attn, 'local_featemb_mut_attn'),
                  (global_featemb_init_attn, 'global_featemb_init_attn'),
                  (global_featemb_mut_attn, 'global_featemb_mut_attn'),
                  (seqlevel_featembeder, 'seqlevel_featembeder'),
                  (decoder, 'decoder'))

        return models

    def _load_model_statedict_(self, models, model_dir):
        print('--- loading trained model ---')
        device = self.device
        fdtype = self.fdtype
        # load state_dict pth
        state_dict_dir = os.path.join(model_dir, 'model_statedict')
        for m, m_name in models:
            m.load_state_dict(torch.load(os.path.join(state_dict_dir, '{}.pkl'.format(m_name)), map_location=device))
        # update model's fdtype and move to device
        for m, m_name in models:
            m.type(fdtype).to(device)
            m.eval()
        return models

    # for updating the embedding matrices to support blank tokens
    # def _load_model_statedict_(self, models, model_dir):
    #     print('--- loading trained model ---')
    #     device = self.device
    #     fdtype = self.fdtype
    #     # load state_dict pth
    #     state_dict_dir = os.path.join(model_dir, 'model_statedict')
    #     for m, m_name in models:
    #         m.load_state_dict(torch.load(os.path.join(state_dict_dir, '{}.pkl'.format(m_name)), map_location=device))
    #         if m_name == 'init_annot_embed':
    #             print(f'--- {m_name} ---')
    #             print('--- updating embedding tracks ---')
    #             # update embedding matrices to support blank token
    #             # updating Wproto
    #             annot_embed = m.Wproto.weight.shape[1]
    #             num_indic = m.num_inidc
    #             emb_ext = torch.nn.Embedding(num_indic + 1, 
    #                                          annot_embed, 
    #                                          padding_idx=num_indic)
    #             emb_ext.weight[:num_indic,:] = m.Wproto.weight
    #             m.Wproto = emb_ext
    #             print('Wproto.shape:', m.Wproto.weight.shape)
    #         if m_name in {'init_annot_embed', 'mut_annot_embed'}:
    #             print(f'--- {m_name} ---')
    #             print('--- updating embedding tracks ---')
    #             num_indic = m.num_inidc
    #             # updating Wpbs
    #             annot_embed = m.Wpbs.weight.shape[1]
    #             emb_ext = torch.nn.Embedding(num_indic + 1, 
    #                                          annot_embed, 
    #                                          padding_idx=num_indic)
    #             emb_ext.weight[:num_indic,:] = m.Wpbs.weight
    #             m.Wpbs = emb_ext
    #             print('Wpbs.shape:', m.Wpbs.weight.shape)

    #             # updating Wrt
    #             annot_embed = m.Wrt.weight.shape[1]
    #             emb_ext = torch.nn.Embedding(num_indic + 1, 
    #                                          annot_embed, 
    #                                          padding_idx=num_indic)
    #             emb_ext.weight[:num_indic,:] = m.Wrt.weight
    #             m.Wrt= emb_ext
    #             print('Wrt.shape:', m.Wrt.weight.shape)
    #         if m_name == 'decoder':
    #             # delete W_sigma
    #             print('--- deleting W_sigma ---')
    #             del m.W_sigma


    #     # update model's fdtype and move to device
    #     for m, m_name in models:
    #         m.type(fdtype).to(device)
    #         m.eval()
    #     return models

    def _run_prediction(self, models, dloader, y_ref=[]):

        device = self.device
        fdtype = self.fdtype
        mask_gen = MaskGenerator()
        requires_grad = False
        pred_score = []

        assert len(y_ref) == 1, 'model predicts one outcome at a time that need to be specified in y_ref! '

        if dloader.dataset.y_score is not None:
            ref_score = []
        else:
            ref_score = None

        seqs_ids_lst = []

        # models = ((init_annot_embed, 'init_annot_embed'), 
        #           (mut_annot_embed, 'mut_annot_embed'),
        #           (init_encoder, 'init_encoder'),
        #           (mut_encoder, 'mut_encoder'),
        #           (local_featemb_init_attn, 'local_featemb_init_attn'),
        #           (local_featemb_mut_attn, 'local_featemb_mut_attn'),
        #           (global_featemb_init_attn, 'global_featemb_init_attn'),
        #           (global_featemb_mut_attn, 'global_featemb_mut_attn'),
        #           (seqlevel_featembeder, 'seqlevel_featembeder'),
        #           (decoder, 'decoder'))

        init_annot_embed = models[0][0]
        mut_annot_embed = models[1][0]
        init_encoder = models[2][0]
        mut_encoder = models[3][0]
        local_featemb_init_attn = models[4][0]
        local_featemb_mut_attn = models[5][0]
        global_featemb_init_attn = models[6][0]
        global_featemb_mut_attn = models[7][0]
        seqlevel_featembeder = models[8][0]
        decoder = models[9][0]


        # going over batches
        for indx_batch, sbatch in tqdm(enumerate(dloader)):
            # print('batch indx:', indx_batch)

            X_init_nucl, X_init_proto, X_init_pbs, X_init_rt, \
            X_mut_nucl, X_mut_pbs, X_mut_rt, \
            x_init_len, x_mut_len, seqlevel_feat, \
            y_val, b_seqs_indx, b_seqs_id = sbatch


            X_init_nucl = X_init_nucl.to(device)

            X_init_proto = X_init_proto.to(device)
            X_init_pbs = X_init_pbs.to(device)
            X_init_rt = X_init_rt.to(device)

            X_mut_nucl = X_mut_nucl.to(device)
            X_mut_pbs = X_mut_pbs.to(device)
            X_mut_rt = X_mut_rt.to(device)
            seqlevel_feat = seqlevel_feat.type(fdtype).to(device)
            # print('seqlevel_feat.shape:', seqlevel_feat.shape)
            # print('seqlevel_feat[0]:', seqlevel_feat[0])

            with torch.set_grad_enabled(False):
                X_init_batch = init_annot_embed(X_init_nucl, X_init_proto, X_init_pbs, X_init_rt)
                X_mut_batch = mut_annot_embed(X_mut_nucl, X_mut_pbs, X_mut_rt)
                # print('X_init_batch.shape:', X_init_batch.shape)
                # print('X_mut_batch.shape:',X_mut_batch.shape)
                # print('x_init_len.shape', x_init_len.shape)
                # print('x_mut_len.shape:', x_mut_len.shape)

                # print(np.unique(x_init_len))
                # print(np.unique(x_mut_len))
                # (bsize,)
 
                if ref_score is not None:
                    # print(y_batch.shape)
                    # print(y_batch.unique())
                    y_batch = y_val.type(fdtype)
                    ref_score.extend(y_batch.view(-1).tolist())
                
                # masks (bsize, init_seqlen) or (bsize, mut_seqlen)
                x_init_m = mask_gen.create_content_mask((X_init_batch.shape[0], X_init_batch.shape[1]), x_init_len)
                x_mut_m =  mask_gen.create_content_mask((X_mut_batch.shape[0], X_mut_batch.shape[1]), x_mut_len)

                __, z_init = init_encoder.forward_complete(X_init_batch, x_init_len, requires_grad=requires_grad)
                __, z_mut =  mut_encoder.forward_complete(X_mut_batch, x_mut_len, requires_grad=requires_grad)
    
                max_seg_len = z_init.shape[1]
                init_mask = x_init_m[:,:max_seg_len].to(device)
                # global attention
                # s (bsize, embed_dim)
                s_init_global, __ = global_featemb_init_attn(z_init, mask=init_mask)
                # local attention
                s_init_local, __ = local_featemb_init_attn(z_init, mask=X_init_rt[:,:max_seg_len])

                max_seg_len = z_mut.shape[1]
                mut_mask = x_mut_m[:,:max_seg_len].to(device)
                s_mut_global, __ = global_featemb_mut_attn(z_mut, mask=mut_mask)
                s_mut_local, __ = local_featemb_mut_attn(z_mut, mask=X_mut_rt[:,:max_seg_len])

                seqfeat = seqlevel_featembeder(seqlevel_feat)
                # y (bsize, 1)
                y_hat_logit = decoder(torch.cat([s_init_global, s_init_local, s_mut_global, s_mut_local, seqfeat], axis=-1))
                
                pred_score.extend(y_hat_logit.view(-1).tolist())
                seqs_ids_lst.extend(list(b_seqs_id))
 
        predictions_df = build_predictions_df(seqs_ids_lst, ref_score, pred_score, y_ref)
        return predictions_df


    def _get_gradients_from_embeddinglayer(self, models, dloader, m_steps=50, num_baseline=1, feat_subs_opt='min_val', y_ref=[]):

        device = self.device
        fdtype = self.fdtype
        mask_gen = MaskGenerator()
        requires_grad = True
        pred_score = []

        assert len(y_ref) == 1, 'model predicts one outcome!'
        
        if dloader.dataset.y_score is not None:
            ref_score = []
        else:
            ref_score = None

        seqs_ids_lst = []

        x_init_len_lst = []
        x_mut_len_lst = []

        ig_init_contrib_lst = []
        ig_mut_contrib_lst = []
        ig_seqfeat_contrib_lst = []
        convg_score_lst = []
        # models = ((init_annot_embed, 'init_annot_embed'), 
        #           (mut_annot_embed, 'mut_annot_embed'),
        #           (init_encoder, 'init_encoder'),
        #           (mut_encoder, 'mut_encoder'),
        #           (local_featemb_init_attn, 'local_featemb_init_attn'),
        #           (local_featemb_mut_attn, 'local_featemb_mut_attn'),
        #           (global_featemb_init_attn, 'global_featemb_init_attn'),
        #           (global_featemb_mut_attn, 'global_featemb_mut_attn'),
        #           (seqlevel_featembeder, 'seqlevel_featembeder'),
        #           (decoder, 'decoder'))

        for m, m_name in models:
            # m.train()
            # disable dropout
            # note: we have to put rnn in training mode so that we can call .backward() later on!
            switch_layer_to_traineval_mode(m, torch.nn.GRU, activate_train=True)
            if m_name in {'init_encoder', 'mut_encoder'}:
                m.rnn.dropout = 0 # enforce no dropout
                print(m_name)
                print(m.rnn_pdropout)
                print(m.rnn.dropout)
            switch_layer_to_traineval_mode(m, torch.nn.Dropout, activate_train=False)
            switch_layer_to_traineval_mode(m, torch.nn.LayerNorm, activate_train=False)

        init_annot_embed = models[0][0]
        mut_annot_embed = models[1][0]
        init_encoder = models[2][0]
        mut_encoder = models[3][0]
        local_featemb_init_attn = models[4][0]
        local_featemb_mut_attn = models[5][0]
        global_featemb_init_attn = models[6][0]
        global_featemb_mut_attn = models[7][0]
        seqlevel_featembeder = models[8][0]
        decoder = models[9][0]
        

        alpha = torch.linspace(start=0., end=1.0, steps=m_steps)
        alpha = alpha.to(device)

        atol = 1e-8 # absolute tolerance to use when using torch.allclose()

        # looping over batches
        # default we process one batch that has
        #  - one reference example as the first row
        #  - corresponding baseline examples below
        topk = num_baseline//2
        if topk == 0:
            topk=1

        annot_blank_token = 2 # 0, 2
        # based on experimental data distribution 
        if feat_subs_opt == 'min_val':
            feat_val_arr = np.array([[ 0.02      ,  0.        ,  0.        ,  0.        ,  0.02      ,
                                       0.06      ,  0.08      ,  0.26      , -0.11583333, -0.44416666,
                                      -0.22166667, -0.52583332, -0.74583333, -0.18999999, -0.055     ,
                                       0.04      ,  0.03      ,  0.13      ,  0.22      ,  0.18      ,
                                       0.        ,  0.        ,  0.        ,  0.        ]])
        elif feat_subs_opt == 'mean_val':
            feat_val_arr = np.array([ 0.02876886,  0.06626062,  0.30763395,  0.62610543,  0.02217379,
                                      0.17397369,  0.33210595,  0.26      , -0.01507402, -0.30927307,
                                     -0.03474034, -0.33915078, -0.53745106, -0.01018871, -0.00220591,
                                      0.2589881 ,  0.13501203,  0.19645158,  0.30725496,  0.45543968,
                                      0.01088532,  0.02019323,  0.30763395,  0.06626062])

        for indx_batch, sbatch in tqdm(enumerate(dloader)):
            # print('batch indx:', indx_batch)
            # for m, m_name in models:
            #     m.zero_grad()
            X_init_nucl, X_init_proto, X_init_pbs, X_init_rt, \
            X_mut_nucl, X_mut_pbs, X_mut_rt, \
            x_init_len, x_mut_len, seqlevel_feat, \
            y_val, b_seqs_indx, b_seqs_id = sbatch

            # print(b_seqs_id)

            X_init_nucl = X_init_nucl.to(device)
            # intervene here and assign blank token to annotations

            X_init_proto[1:,:] = annot_blank_token
            X_init_pbs[1:,:] = annot_blank_token
            X_init_rt[1:,:] = annot_blank_token

            X_init_proto = X_init_proto.to(device)
            X_init_pbs = X_init_pbs.to(device)
            X_init_rt = X_init_rt.to(device)


            X_mut_nucl = X_mut_nucl.to(device)
            # intervene here and assign blank token to annotations
            X_mut_pbs[1:,:] = annot_blank_token
            X_mut_rt[1:,:] = annot_blank_token
            X_mut_pbs = X_mut_pbs.to(device)
            X_mut_rt = X_mut_rt.to(device)


            # print(X_init_nucl)
            # print(X_mut_nucl)

            # [bsize, featuredim]
            seqlevel_feat = seqlevel_feat.type(fdtype).to(device)
            seqlevel_feat.requires_grad=requires_grad

            with torch.set_grad_enabled(requires_grad):
                X_init_batch = init_annot_embed(X_init_nucl, X_init_proto, X_init_pbs, X_init_rt)
                X_mut_batch = mut_annot_embed(X_mut_nucl, X_mut_pbs, X_mut_rt)
                
                # require_nonleaf_grad(X_init_batch)
                # require_nonleaf_grad(X_mut_batch)
                #TODO: run on the reference sequence to get prediction score and subtract when computing
                # convergence score

                # [1, maxseqlen, embed_dim]
                X_ref_init_batch = X_init_batch[0].unsqueeze(0)
                X_ref_mut_batch = X_mut_batch[0].unsqueeze(0)
                
                # print('X_ref_init_batch.requires_grad:',X_ref_init_batch.requires_grad)
                # print('X_ref_mut_batch.requires_grad:',X_ref_mut_batch.requires_grad)
                
                # [num_baselines, maxseqlen, embed_dim]
                X_base_init_batch = X_init_batch[1:]
                X_base_mut_batch = X_mut_batch[1:]

                num_baselines =  X_base_init_batch.shape[0]
                max_seqlen = X_base_init_batch.shape[1]
                alpha_exp = get_alpha_expanded(alpha, num_baselines, max_seqlen)
                # print('alpha_exp.shape:', alpha_exp.shape)
                # print('alpha_exp.device:', alpha_exp.device)

                # [1, m_steps, maxseqlen, embed_dim]
                X_ref_init_batch_exp = expand_tensor_msteps(X_ref_init_batch, m_steps)
                # print('X_ref_init_batch_exp.device:', X_ref_init_batch_exp.device)
                # [num_baselines, m_steps, maxseqlen, embed_dim]
                X_base_init_batch_exp = expand_tensor_msteps(X_base_init_batch, m_steps)
                # print('X_base_init_batch_exp.device:',X_base_init_batch_exp.device)
                # [num_baselines, m_steps, maxseqlen, embed_dim]
                dist_init = X_ref_init_batch_exp - X_base_init_batch_exp
                X_init_batch_interpolated = X_base_init_batch_exp + alpha_exp * dist_init
                # assert torch.allclose(X_init_batch_interpolated[:,0,:,:],X_base_init_batch)
                # for nb in range(num_baselines):
                #     assert torch.allclose(X_init_batch_interpolated[nb,-1,:,:], X_ref_init_batch)
                # require_nonleaf_grad(X_init_batch_interpolated)
                # print('X_init_batch_interpolated:', X_init_batch_interpolated.shape)
                # print('X_init_batch_interpolated.grad:', X_init_batch_interpolated.requires_grad)


                alpha_exp = get_alpha_expanded(alpha, num_baselines, max_seqlen)
                # [1, m_steps, maxseqlen, embed_dim]
                X_ref_mut_batch_exp = expand_tensor_msteps(X_ref_mut_batch, m_steps)
                # [num_baselines, m_steps, maxseqlen, embed_dim]
                X_base_mut_batch_exp = expand_tensor_msteps(X_base_mut_batch, m_steps)
                dist_mut = X_ref_mut_batch_exp - X_base_mut_batch_exp
                X_mut_batch_interpolated = X_base_mut_batch_exp + alpha_exp * dist_mut
                # assert torch.allclose(X_mut_batch_interpolated[:,0,:,:],X_base_mut_batch)
                # for nb in range(num_baselines):
                #     assert torch.allclose(X_mut_batch_interpolated[nb,-1,:,:], X_ref_mut_batch)
                # require_nonleaf_grad(X_mut_batch_interpolated)
                # print('X_mut_batch_interpolated:', X_mut_batch_interpolated.shape)
                # print('X_mut_batch_interpolated.grad:', X_mut_batch_interpolated.requires_grad)


                # [1, 1, seqlevel_featdim]
                ref_seqfeat = seqlevel_feat[0].unsqueeze(0).unsqueeze(0)
                # print('ref_seqfeat.shape:',ref_seqfeat.shape)
                # zero the baselines (another option is to uniform sample from the range)
                # [num_baselines, 1, seqlevel_featdim]
                base_seqfeat = seqlevel_feat[1:]
                # print('base_seqfeat.shape:', base_seqfeat.shape)
                base_seqfeat = torch.from_numpy(feat_val_arr).type(fdtype).to(device).expand(base_seqfeat.shape[0],-1)
                # print('base_seqfeat.shape:', base_seqfeat.shape)

                base_seqfeat = base_seqfeat.unsqueeze(1)
                # print('base_seqfeat.shape:', base_seqfeat.shape)

                # print(base_seqfeat)
                # base_seqfeat = torch.zeros_like(seqlevel_feat[1:].unsqueeze(1))
                # base_seqfeat = seqlevel_feat[1:].unsqueeze(1)
                # print('base_seqfeat:', base_seqfeat)
                # print('base_seqfeat.shape:',base_seqfeat.shape)

                alpha_exp = get_alpha_expanded(alpha, num_baselines, 1)
                # print('alpha_exp.shape:',alpha_exp.shape)
                 # [1, m_steps, 1, seqlevel_featdim]
                ref_seqfeat_exp = expand_tensor_msteps(ref_seqfeat, m_steps)
                # [num_baselines, m_steps, 1, seqlevel_featdim]
                base_seqfeat_exp = expand_tensor_msteps(base_seqfeat, m_steps)
                dist_seqfeat = ref_seqfeat_exp - base_seqfeat_exp
                seqfeat_interpolated = base_seqfeat_exp +  alpha_exp * dist_seqfeat
                # assert torch.allclose(seqfeat_interpolated[:,0,:,:],base_seqfeat)
                # for nb in range(num_baselines):
                #     assert torch.allclose(seqfeat_interpolated[nb,-1,:,:], ref_seqfeat)
                # require_nonleaf_grad(ref_seqfeat_interpolated)
                # print('seqfeat_interpolated:', seqfeat_interpolated.shape)
                # print('seqfeat_interpolated.grad:', seqfeat_interpolated.requires_grad)

                # (bsize,)
                if ref_score is not None:
                    y_batch = y_val.type(fdtype).to(device)
                    ref_score.append(y_batch[0,0].item())
                
                # print('x_init_len:', x_init_len.shape)
                # print('x_init_len:', x_init_len)
                # print('x_mut_len.shape:', x_mut_len.shape)
                # print('x_mut_len:', x_mut_len)

                # print('X_init_batch.shape:', X_init_batch.shape)

                # keep track of sequence lengths
                x_init_len_lst.append(x_init_len[0].item())
                x_mut_len_lst.append(x_mut_len[0].item())

                x_init_len_interp = torch.full((num_baselines*m_steps,), x_init_len[0].item(), device=device)
                x_mut_len_interp = torch.full((num_baselines*m_steps,), x_mut_len[0].item(), device=device)
                
                # print('x_init_len_interp.shape:', x_init_len_interp.shape)

                # [num_baselines*m_steps, maxseqlen, embed_dim]
                X_init_batch_interpolated = X_init_batch_interpolated.reshape(-1, 
                                                                              X_init_batch_interpolated.shape[-2],
                                                                              X_init_batch_interpolated.shape[-1])

                X_mut_batch_interpolated = X_mut_batch_interpolated.reshape(-1, 
                                                                              X_mut_batch_interpolated.shape[-2],
                                                                              X_mut_batch_interpolated.shape[-1])
                require_nonleaf_grad(X_init_batch_interpolated, 'X_init_batch_interpolated')
                require_nonleaf_grad(X_mut_batch_interpolated, 'X_mut_batch_interpolated')
                # print('X_init_batch_interpolated.shape:', X_init_batch_interpolated.shape)
                # print('X_mut_batch_interpolated.shape:', X_mut_batch_interpolated.shape)
                # masks (bsize, init_seqlen) or (bsize, mut_seqlen)
                x_init_m = mask_gen.create_content_mask((X_init_batch_interpolated.shape[0], 
                                                         X_init_batch_interpolated.shape[1]), 
                                                         x_init_len_interp)
                x_mut_m =  mask_gen.create_content_mask((X_mut_batch_interpolated.shape[0], 
                                                         X_mut_batch_interpolated.shape[1]), 
                                                         x_mut_len_interp)

                __, z_init = init_encoder.forward_complete(X_init_batch_interpolated, x_init_len_interp, requires_grad=requires_grad)
                __, z_mut =  mut_encoder.forward_complete(X_mut_batch_interpolated, x_mut_len_interp, requires_grad=requires_grad)
    
                max_seg_len = z_init.shape[1]
                init_mask = x_init_m[:,:max_seg_len].to(device)

                # print('z_init.shape:', z_init.shape)
                # print('z_mut.shape:', z_mut.shape)
                # print('init_mask.shape:', init_mask.shape)
                # print('X_init_rt.shape:',X_init_rt.shape)
                # print('max_seg_len:', max_seg_len)
                # print('X_init_rt:', X_init_rt)

                init_rt_mask = X_init_rt[0].unsqueeze(0).expand(num_baselines*m_steps,-1)
                # global attention
                # s (bsize, embed_dim)
                s_init_global, __ = global_featemb_init_attn(z_init, mask=init_mask)
                # local attention
                s_init_local, __ = local_featemb_init_attn(z_init, mask=init_rt_mask[:,:max_seg_len])

                max_seg_len = z_mut.shape[1]
                mut_mask = x_mut_m[:,:max_seg_len].to(device)
                s_mut_global, __ = global_featemb_mut_attn(z_mut, mask=mut_mask)
                mut_rt_mask = X_mut_rt[0].unsqueeze(0).expand(num_baselines*m_steps,-1)
                s_mut_local, __ = local_featemb_mut_attn(z_mut, mask=mut_rt_mask[:,:max_seg_len])
                
                seqfeat_interpolated = seqfeat_interpolated.reshape(-1, seqfeat_interpolated.shape[-1])
                require_nonleaf_grad(seqfeat_interpolated, 'seqfeat_interpolated')

                # print('s_init_global.shape:',s_init_global.shape)
                # print('s_init_local.shape:',s_init_local.shape)
                # print('s_mut_global.shape:', s_mut_global.shape)
                # print('s_mut_local.shape:', s_mut_local.shape)
                # print('seqfeat_interpolated:', seqfeat_interpolated.shape)

                z_seqfeat = seqlevel_featembeder(seqfeat_interpolated)
                # print('z_seqfeat.shape:', z_seqfeat.shape)
                # y (bsize, 1)
                y_hat_logit = decoder(torch.cat([s_init_global, s_init_local, s_mut_global, s_mut_local, z_seqfeat], axis=-1))
                
                # print('y_hat_logit.shape:',y_hat_logit.shape)
                # print('y_batch.shape:',y_batch.shape)

                # y_batch_interp = y_batch[0].unsqueeze(0).expand(num_baselines*m_steps,-1)
                # print('y_batch_interp.shape:',y_batch_interp.shape)
                # print('y_hat_logit.shape:', y_hat_logit.shape)
                
                # f_xref = y_batch[0].unsqueeze(0).expand(num_baselines,-1)
                # print(y_hat_logit.reshape(num_baselines, m_steps, -1))
                f_xref = y_hat_logit.reshape(num_baselines, m_steps, -1)[:,-1,:]
                f_xbase = y_hat_logit.reshape(num_baselines, m_steps, -1)[:,0,:]
                # print('f_xref:', f_xref)
                # print('f_xbase:', f_xbase)
                pred_score.append(f_xref[0,0].item())

                # print('X_init_batch_interpolated.grad:', X_init_batch_interpolated.grad)
                # print('X_mut_batch_interpolated.grad:', X_mut_batch_interpolated.grad)
                # print('seqfeat_interpolated.grad:', seqfeat_interpolated.grad)

                # y_hat_logit.sum().backward(retain_graph=True)
                y_hat_logit.backward(torch.ones(y_hat_logit.size(), device=device),retain_graph=True)
                
                # a_opt1 = X_init_batch_interpolated.grad.clone()
                # b_opt1 = X_mut_batch_interpolated.grad.clone()
                # c_opt1 = seqfeat_interpolated.grad.clone()
                # for i in range(y_hat_logit.shape[0]):
                #     # a is a tuple of length equal to 1 that has the gradient
                #     # gradient is [num_samples, seqlen, embed_dim]
                #     a = torch.autograd.grad(y_hat_logit[i,0], X_init_batch_interpolated, retain_graph=True)
                #     # print('a.shape', a.shape)
                #     # print('i:', i)
                #     # print(a[0].shape)
                #     # print(a_opt1.shape)
                #     # print(a[0][i])
                #     # print(a_opt1[i])
                    
                #     assert torch.allclose(a[0][i], a_opt1[i])
                #     print('a[0].sum():',a[0].sum(), f'a[0][{i}].sum():',a[0][i].sum())
                #     print((a[0].sum()-a[0][i].sum()).abs())
                #     assert torch.allclose(a[0].sum(),a[0][i].sum(), atol=atol) # test if the gradients are not accumulating
                #     # print()

                #     b = torch.autograd.grad(y_hat_logit[i,0], X_mut_batch_interpolated, retain_graph=True)
                #     assert torch.allclose(b[0][i], b_opt1[i])
                #     assert torch.allclose(b[0].sum(),b[0][i].sum(), atol=atol) # test if the gradients are not accumulating
                #     # print()
                #     c = torch.autograd.grad(y_hat_logit[i,0], seqfeat_interpolated, retain_graph=True)
                #     assert torch.allclose(c[0][i], c_opt1[i])
                #     assert torch.allclose(c[0].sum(),c[0][i].sum(), atol=atol) # test if the gradients are not accumulating
                #     # print()    
                # print('assert gradient equivalence')      
                # assert torch.allclose( X_init_batch_interpolated.grad, a_opt1)
                # assert torch.allclose( X_mut_batch_interpolated.grad, b_opt1)
                # assert torch.allclose( seqfeat_interpolated.grad, c_opt1)
                # for o in range(y_hat_logit.shape[0]):
                #     y_hat_logit[o].backward(retain_graph=True)

                # print()
                # print('y_batch_interp:',y_batch_interp)
                # print('y_logit_hat:', y_hat_logit)
                # loss = loss_func(y_batch_interp, y_hat_logit)
                # loss.backward()
                #TODO: add diagnositcs plots such as 
                # plot of alpha by average gradients 
                # add function to do sanity check on the computation of alpha
                # print('X_init_batch_interpolated.grad.shape:',X_init_batch_interpolated.grad.shape)
                
                # [num_baselines, m_steps, maxseqlen, embed_dim]
                init_grad = X_init_batch_interpolated.grad.reshape(num_baselines,
                                                              m_steps, 
                                                              X_init_batch_interpolated.shape[-2], 
                                                              X_init_batch_interpolated.shape[-1])
                mut_grad = X_mut_batch_interpolated.grad.reshape(num_baselines,
                                                            m_steps,
                                                            X_mut_batch_interpolated.shape[-2],
                                                            X_mut_batch_interpolated.shape[-1])
                seqfeat_grad = seqfeat_interpolated.grad.reshape(num_baselines, m_steps, -1)
                # print()
                # print('init_grad.shape:',init_grad.shape)
                # print('mut_grad.shape:',mut_grad.shape)
                # print('seqfeat_grad.shape:',seqfeat_grad.shape)
                 
                # [num_baselines, max_seqlen, embedd_dim]
                ig_init_grad = self.compute_riemann_trapezoidal_approximation(init_grad)
                ig_mut_grad = self.compute_riemann_trapezoidal_approximation(mut_grad)
                # [num_baselines, seqlevelfeat_dim]
                ig_seqfeat_grad = self.compute_riemann_trapezoidal_approximation(seqfeat_grad)
                
                # print()
                # print('ig_init_grad.shape:',ig_init_grad.shape)
                # print('ig_mut_grad.shape:',ig_mut_grad.shape)
                # print('ig_seqfeat_grad.shape:',ig_seqfeat_grad.shape)
                
                # TODO: figure out to only keep the actual length 
                # it won't matter for the computation as we are not summing or avg. on the sequence len dimension
                # print(ig_init_grad.mean(axis=0).shape)
                # print(ig_init_grad.mean(axis=0)[:x_init_len[0].item()].shape)
                #[num_baselines, max_seqlen, embed_dim]
                ig_init_grad_scaled = (X_ref_init_batch - X_base_init_batch)*ig_init_grad
                ig_mut_grad_scaled = (X_ref_mut_batch - X_base_mut_batch)*ig_mut_grad
                # [num_baselines, seqlevelfeat_dim]
                ig_seqfeat_grad_scaled = (ref_seqfeat - base_seqfeat).squeeze(-2) * ig_seqfeat_grad
                # print()
                # print('ig_init_grad_scaled.shape:',ig_init_grad_scaled.shape)
                # print('ig_mut_grad_scaled.shape:',ig_mut_grad_scaled.shape)
                # print('ig_seqfeat_grad_scaled.shape:',ig_seqfeat_grad_scaled.shape)
                
                # print()
                # print('ig grad sum')
                # print(ig_init_grad.sum())
                # print(ig_mut_grad.sum())
                # print(ig_seqfeat_grad.sum())
                # print('ig grad abs. sum')
                # print(ig_init_grad.abs().sum())
                # print(ig_mut_grad.abs().sum())
                # print(ig_seqfeat_grad.abs().sum())

                # print('ig grad scaled sum')
                # print(ig_init_grad_scaled.sum())
                # print(ig_mut_grad_scaled.sum())
                # print(ig_seqfeat_grad_scaled.sum())
                # print('ig grad scaled abs. sum')
                # print(ig_init_grad_scaled.abs().sum())
                # print(ig_mut_grad_scaled.abs().sum())
                # print(ig_seqfeat_grad_scaled.abs().sum())

                # print('-'*10)

                init_len = x_init_len[0].item()
                mut_len  = x_mut_len[0].item()
                sum_attr_ = self.sum_ig_contrib(ig_init_grad_scaled,init_len) + \
                            self.sum_ig_contrib(ig_mut_grad_scaled, mut_len) + \
                            ig_seqfeat_grad_scaled.sum(axis=-1)
                # print('f_xref.shape:', f_xref.shape)
                # print('f_xref', f_xref)
                # print('f_base.shape:', f_xbase.shape)
                # print('f_xbase:', f_xbase)
                fsum_ = (f_xref-f_xbase).squeeze(-1)
                # print('sum_attr ig_grad_scaled:', sum_attr_)
                # print('fsum:', fsum_)
                diff = (sum_attr_-fsum_).abs()
                # print('diff abs:', diff)
                topk_indices = torch.topk(-1*diff,topk).indices
                # print('topk_indices:', topk_indices)
                # print(diff[topk_indices])
                # print(ig_init_grad_scaled[topk_indices].shape)
                # convg_ = (sum_attr_.mean()-fsum_.mean()).abs().item()
                convg_ = diff[topk_indices].mean().item()
                # print('convg:', convg_)

                # sum_attr = ig_init_grad_scaled.abs().sum() + ig_mut_grad_scaled.abs().sum() + ig_seqfeat_grad_scaled.abs().sum()
                # fsum =  num_baselines*(f_xref-f_xbase).sum()
                # print('sum_attr (abs) ig_grad_scaled:', sum_attr.item())
                # print('fsum:', fsum.item())

                # sum_attr = ig_init_grad.abs().sum() + ig_mut_grad.abs().sum() + ig_seqfeat_grad.abs().sum()
                # fsum =  num_baselines*(f_xref-f_xbase).sum()
                # print('sum_attr (abs) ig_grad:', sum_attr.item())
                # print('fsum:', fsum.item())

                # (max_seqlen)
                ig_init_contrib = self.compute_avgbaseline_ig(ig_init_grad_scaled[topk_indices])
                # (max_seqlen)
                ig_mut_contrib = self.compute_avgbaseline_ig(ig_mut_grad_scaled[topk_indices])
                # (seqlevel_featuredim)
                ig_seqfeat_contrib = self.compute_avgbaseline_ig(ig_seqfeat_grad_scaled[topk_indices], sum_along_featdim=False)

                ig_init_contrib_lst.append(ig_init_contrib.detach().cpu().numpy())
                ig_mut_contrib_lst.append(ig_mut_contrib.detach().cpu().numpy())
                ig_seqfeat_contrib_lst.append(ig_seqfeat_contrib.detach().cpu().numpy())

                # print()
                # print('ig_init_contrib.shape:',ig_init_contrib.shape)
                # print('ig_mut_contrib.shape:',ig_mut_contrib.shape)
                # print('ig_seqfeat_contrib.shape:',ig_seqfeat_contrib.shape)
   

                # sum_attr_avg = ig_init_contrib[:init_len].sum() + \
                #                ig_mut_contrib[:mut_len].sum() + \
                #                ig_seqfeat_contrib.sum()

                # fsum_avg = (f_xref.mean()-f_xbase.mean())
                # print('sum_attr avg ig_grad_scaled:', sum_attr_avg)
                # print('fsum avg:', fsum_avg)
                # convg_avg = (sum_attr_avg-fsum_avg).abs().item()
                # print('convg:', convg_avg)

                # print('-'*10)
                # print('avg. ig')
                # print(ig_init_contrib.sum())
                # print(ig_mut_contrib.sum())
                # print(ig_seqfeat_contrib.sum())
                # print()
                
                # print('avg. ig abs.')
                # print(ig_init_contrib.abs().sum())
                # print(ig_mut_contrib.abs().sum())
                # print(ig_seqfeat_contrib.abs().sum())
                # print()


                # convg_score_lst.append([convg_, convg_avg])
                # print('convg score: ', convg_score_lst)
                # seqs_ids_lst.append(b_seqs_id[0])


                convg_score_lst.append([convg_])
                # print('convg score: ', convg_score_lst)
                seqs_ids_lst.append(b_seqs_id[0])
 
        return seqs_ids_lst, pred_score, ref_score, x_init_len_lst, x_mut_len_lst, ig_init_contrib_lst, ig_mut_contrib_lst, ig_seqfeat_contrib_lst, convg_score_lst
    
    def sum_ig_contrib(self, ig_tensor, seqlen):
        """
        Args:
            ig_tensor: torch tensor, [num_baselines, max_seqlen, featdim]
            seqlen: int, length of the sequence
        Returns:
            tensor [num_baselines]
        
        """
        return ig_tensor.sum(axis=-1)[:, :seqlen].sum(axis=-1)

    def compute_avgbaseline_ig(self, ig_tensor, sum_along_featdim=True):
        """
        Args:
            ig_tensor: torch tensor representing integrated gradients
                       [num_baselines, max_seqlen, featdim]
        Returns:
            tensor [max_seqlen] or [seqlevelfeat_dim]
        
        """
        if sum_along_featdim:
            return ig_tensor.mean(axis=0).sum(axis=-1)
        else:
            return ig_tensor.mean(axis=0)

    def compute_riemann_trapezoidal_approximation(self, gradients):
        """
        Args:
            gradients: tensor, (num_baselines, m_steps, maxseqlen, embed_dim)
        Returns:
            integrated gradient: tensor (num_baselines, maxseqlen, embed_dim)
        """
        # sum across the m_steps dimension
        grads = (gradients[:,:-1] + gradients[:,1:]) / 2.
        # average across m_steps dimension
        integrated_gradients = torch.mean(grads, axis=1)
        return integrated_gradients

    def prepare_data(self, df, maskpos_indices=None, y_ref=[], batch_size=500):
        """
        Args:
            df: pandas dataframe
            y_ref: list (optional), list of reference outcome names
            batch_size: int, number of samples to process per batch
        """
        norm_colnames, df, proc_seq_init_df,num_init_cols, proc_seq_mut_df, num_mut_cols =  self._process_df(df)
        dtensor = self._construct_datatensor(norm_colnames, 
                                             df, 
                                             proc_seq_init_df,
                                             num_init_cols, 
                                             proc_seq_mut_df, 
                                             num_mut_cols, 
                                             maskpos_indices=maskpos_indices,
                                             y_ref=y_ref)
        dloader = self._construct_dloader(dtensor, batch_size)
        return dloader


    def compute_integratedgradients_workflow(self, dloader, model_dir, m_steps=50, num_baseline=1, feat_subs_opt='min_val', y_ref=[]):

        mconfig_dir = os.path.join(model_dir, 'config')
        mconfig = self._load_model_config(mconfig_dir)
        
        # list of tuples (model, model_name)
        models = self._build_base_model(mconfig)

        models = self._load_model_statedict_(models, model_dir)
        # print(f'running prediction for prime editor | model_dir: {model_dir}')

        return self._get_gradients_from_embeddinglayer(models,
                                                       dloader, 
                                                       m_steps=m_steps,
                                                       num_baseline=num_baseline,
                                                       feat_subs_opt=feat_subs_opt,
                                                       y_ref=y_ref)


    def build_retrieve_models(self, model_dir):

        mconfig_dir = os.path.join(model_dir, 'config')
        mconfig = self._load_model_config(mconfig_dir)
        
        # list of tuples (model, model_name)
        models = self._build_base_model(mconfig)

        models = self._load_model_statedict_(models, model_dir)


        return models

    def predict_from_dloader_knockoff(self, dloader, model_dir, seqlevelfeat_indices, y_ref=[]):

        mconfig_dir = os.path.join(model_dir, 'config')
        mconfig = self._load_model_config(mconfig_dir)
        
        # list of tuples (model, model_name)
        models = self._build_base_model(mconfig)
        models = self._load_model_statedict_(models, model_dir)
        # print(f'running prediction for prime editor | model_dir: {model_dir}')
        ## Assign 0 to weights used for mapping seqlevel_feature to embedding vector
        if seqlevelfeat_indices is not None:
            fdtype = self.fdtype
            device = self.device
            with torch.no_grad():
                seqembeder_weight = models[-2][0].We.weight.clone()
                # print('original weights:')
                # print(seqembeder_weight[:, seqlevelfeat_indices].sum())
                mask_val = torch.zeros(seqembeder_weight.shape[0], len(seqlevelfeat_indices)).type(fdtype).to(device)
                # mask target values
                models[-2][0].We.weight[:, seqlevelfeat_indices] = mask_val
                # print('masked weights:')
                # print(models[-2][0].We.weight[:,seqlevelfeat_indices].sum())

        pred_df = self._run_prediction(models, dloader, y_ref=y_ref)

        return pred_df

    def predict_from_dloader(self, dloader, model_dir, y_ref=[]):

        mconfig_dir = os.path.join(model_dir, 'config')
        mconfig = self._load_model_config(mconfig_dir)
        
        # list of tuples (model, model_name)
        models = self._build_base_model(mconfig)
        models = self._load_model_statedict_(models, model_dir)
        # print(f'running prediction for prime editor | model_dir: {model_dir}')
        pred_df = self._run_prediction(models, dloader, y_ref=y_ref)

        return pred_df

    def compute_avg_predictions(self, df):
        agg_df = df.groupby(by=['seq_id']).mean()
        agg_df.reset_index(inplace=True)
        for colname in ('run_num', 'Unnamed: 0'):
            if colname in agg_df:
                del agg_df[colname]
        return agg_df
    
    def prepare_df_for_viz(self, df):
        norm_colnames, df, proc_seq_init_df,num_init_cols, proc_seq_mut_df, num_mut_cols =  self._process_df(df)
        viz_tup = (proc_seq_init_df, proc_seq_mut_df, max(num_init_cols,num_mut_cols), df)
        return viz_tup

    def visualize_seqs(self, viz_tup, seqsids_lst):
        """
        Args:
            df: pandas dataframe
            seqids_lst: list of sequence ids to plot

        """

        out_tb = {}
        df = viz_tup[-1] # last element
        tseqids = set(seqsids_lst).intersection(set(df['seq_id'].unique()))

        for tseqid in tqdm(tseqids):
            out_tb[tseqid] = Viz_PESeqs().viz_align_initmut_seq(*viz_tup,
                                                                tseqid, 
                                                                window=self.wsize,
                                                                return_type='html')

        return out_tb