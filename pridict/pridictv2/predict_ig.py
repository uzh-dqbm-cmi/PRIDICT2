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
from .feature_importance.ig_explainer import IntegratedGradExplainer


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

    def _set_model_trainmode(self, models):
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
    
    def _embed_tokens(self, embedder, input_args):
        with torch.set_grad_enabled(False):
            X_embed = embedder(*input_args)
        return X_embed

    def _zero_model_grads(self, models):
        for m, m_name in models:
            m.zero_grad()

    def _prepare_baseline_input_batch(self, baseline_input_batch, ig_explainer, n):
        X_init_nucl, X_init_proto, X_init_pbs, X_init_rt, \
        X_mut_nucl, X_mut_pbs, X_mut_rt, \
        x_init_len, x_mut_len, seqlevel_feat, \
        y_val, b_seqs_indx, b_seqs_id = baseline_input_batch

        device = self.device
        X_init_nucl_exp = ig_explainer.expand_baseline(X_init_nucl, n).to(device)
        X_init_proto_exp = ig_explainer.expand_baseline(X_init_proto, n).to(device)
        X_init_pbs_exp = ig_explainer.expand_baseline(X_init_pbs, n).to(device)
        X_init_rt_exp = ig_explainer.expand_baseline(X_init_rt, n).to(device)

        X_mut_nucl_exp = ig_explainer.expand_baseline(X_mut_nucl, n).to(device)
        X_mut_pbs_exp = ig_explainer.expand_baseline(X_mut_pbs, n).to(device)
        X_mut_rt_exp = ig_explainer.expand_baseline(X_mut_rt, n).to(device)

        seqlevel_feat = seqlevel_feat.type(self.fdtype)
        seqlevel_feat_exp = ig_explainer.expand_baseline(seqlevel_feat, n).to(device)

        return [[X_init_nucl_exp, X_init_proto_exp, X_init_pbs_exp, X_init_rt_exp],
                [X_mut_nucl_exp, X_mut_pbs_exp, X_mut_rt_exp],
                seqlevel_feat_exp]

    def _reshape_bsize_k(self, dtensor, bsize, k):
        # dtensor has shape (bsize*k, ...) where ... refers to the rest of the dimensions
        return dtensor.reshape([bsize, k] + list(dtensor.shape[1:]))

    def _get_integrated_gradients(self, models, dloader, bg_set, m_steps=50, y_ref=[]):

        device = self.device
        fdtype = self.fdtype
        mask_gen = MaskGenerator()
        requires_grad = True
        pred_score = []
        outcome_name_indices_map ={'averageedited':0}

        # determine which outcome to use
        assert len(y_ref) == 1, f'# of target outcomes for IG comptation should be 1. Options are: \n{outcome_name_indices_map}'
        outcome_indx = outcome_name_indices_map[y_ref[0]]

        if dloader.dataset.y_score is not None:
            ref_score = []
        else:
            ref_score = None
        
        ig_explainer = IntegratedGradExplainer(bg_set, 
                                               device, 
                                               k=m_steps, 
                                               scale_by_inputs=True, 
                                               fdtype=fdtype)

        print(f'y_ref:{y_ref}, outcome_indx:{outcome_indx}')
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
        self._set_model_trainmode(models)

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
        

        topk = 1
        
        for indx_batch, sbatch in tqdm(enumerate(dloader)):
            # print('batch indx:', indx_batch)
            # clear the gradients for each batch
            self._zero_model_grads(models)

            #######
            # get input samples to explain
            ########
            X_init_nucl, X_init_proto, X_init_pbs, X_init_rt, \
            X_mut_nucl, X_mut_pbs, X_mut_rt, \
            x_init_len, x_mut_len, seqlevel_feat, \
            y_val, b_seqs_indx, b_seqs_id = sbatch


            bsize = X_init_nucl.shape[0]
            # print(b_seqs_id)
            
            # these inputs are mainly tokens and annotations from set of tokens
            X_init_nucl = X_init_nucl.to(device)
            X_init_proto = X_init_proto.to(device)
            X_init_pbs = X_init_pbs.to(device)
            X_init_rt = X_init_rt.to(device)

            X_mut_nucl = X_mut_nucl.to(device)
            X_mut_pbs = X_mut_pbs.to(device)
            X_mut_rt = X_mut_rt.to(device)
            # print(X_init_nucl)
            # print(X_mut_nucl)

            ########
            # embed input samples (from characters/tokens to vectors representation)
            ########

            #[bsize, seqlen, featdim]
            X_init_batch = self._embed_tokens(init_annot_embed, [X_init_nucl, X_init_proto, X_init_pbs, X_init_rt])
            X_mut_batch = self._embed_tokens(mut_annot_embed, [X_mut_nucl, X_mut_pbs, X_mut_rt])
            # [bsize, featuredim]
            seqlevel_feat = seqlevel_feat.type(fdtype).to(device)

            #######
            # get baselines (batch of baseline to use)
            ########
            ig_explainer.create_bgset_sampler(bsize)
            baseline_input_batch = ig_explainer.get_bgset_batch()
            baseline_input_batch_exp_lst = self._prepare_baseline_input_batch(baseline_input_batch, ig_explainer, m_steps)
            
            #######
            # embed baseline samples (from characters/tokens to vectors representation)
            ########
            #[bsize, k, seqlen, featdim]
            X_init_baseline_batch = self._embed_tokens(init_annot_embed, baseline_input_batch_exp_lst[0])
            X_mut_baseline_batch = self._embed_tokens(mut_annot_embed, baseline_input_batch_exp_lst[1])
            #[bsize, k, featdim]
            seqlevel_feat_baseline = baseline_input_batch_exp_lst[2]

            X_init_baseline_batch.requires_grad = requires_grad
            X_mut_baseline_batch.requires_grad = requires_grad
            seqlevel_feat_baseline.requires_grad = requires_grad

            
            # print('seqlevel_feat:', seqlevel_feat)
            # print('seqlevel_feat_baseline:', seqlevel_feat_baseline)
            #########
            # get interpolated input
            #########
            # [bsize, k, seqlen, featdim]
            X_init_interp = ig_explainer.get_samples_input(X_init_batch, X_init_baseline_batch)
            X_mut_interp = ig_explainer.get_samples_input(X_mut_batch, X_mut_baseline_batch)
            # [bsize, k, featdim]
            seqlevel_interp = ig_explainer.get_samples_input(seqlevel_feat, seqlevel_feat_baseline)

            #########
            # get distance (delta) between input and baseline
            #########
            #[bsize, k, seqlen, featdim]
            dist_init = ig_explainer.get_samples_delta(X_init_batch, X_init_baseline_batch)
            dist_mut = ig_explainer.get_samples_delta(X_mut_batch, X_mut_baseline_batch)
            #[bsize, k, featdim]
            dist_seqlevelfeat = ig_explainer.get_samples_delta(seqlevel_feat, seqlevel_feat_baseline)
            

            ###################
            # compute gradients
            ###################
            
            with torch.set_grad_enabled(requires_grad):
                # (bsize,)
                if ref_score is not None:
                    y_batch = y_val.type(fdtype).to(device)
                    #TODO: double check this one if it correctly getting the target outcome
                    # i think we are passing outcome y with dimension 1
                    ref_score.extend(y_batch[:,0].tolist())


                # keep track of sequence lengths
                x_init_len_lst.extend(x_init_len.tolist())
                x_mut_len_lst.extend(x_mut_len.tolist())

                # [num_baselines*m_steps, maxseqlen, embed_dim]
                X_init_interp = X_init_interp.reshape([-1] + list(X_init_interp.shape[-2:]))
                X_mut_interp = X_mut_interp.reshape([-1] + list(X_mut_interp.shape[-2:]))

                x_init_len_interp = torch.repeat_interleave(x_init_len, m_steps, axis=0).to(device)
                x_mut_len_interp = torch.repeat_interleave(x_mut_len, m_steps, axis=0).to(device)
                # print('x_init_len_interp.shape:', x_init_len_interp.shape)

                # print('X_init_interp.shape:', X_init_interp.shape)
                # print('X_mut_interp.shape:', X_mut_interp.shape)

                __, z_init = init_encoder.forward_complete(X_init_interp, x_init_len_interp, requires_grad=requires_grad)
                __, z_mut =  mut_encoder.forward_complete(X_mut_interp, x_mut_len_interp, requires_grad=requires_grad)
    
                # masks (bsize*k, init_seqlen)
                m_shape = (X_init_interp.shape[0], X_init_interp.shape[1])
                x_init_m = mask_gen.create_content_mask(m_shape, x_init_len_interp)
                # RNN api making sure to use the max sequence length in current batch
                max_seg_len = z_init.shape[1]
                init_mask = x_init_m[:,:max_seg_len].to(device)

                # print('z_init.shape:', z_init.shape)
                # print('z_mut.shape:', z_mut.shape)
                # print('init_mask.shape:', init_mask.shape)
                # print('X_init_rt.shape:',X_init_rt.shape)
                # print('max_seg_len:', max_seg_len)
                # print('X_init_rt:', X_init_rt)

                # global attention
                # s (bsize, embed_dim)
                s_init_global, __ = global_featemb_init_attn(z_init, mask=init_mask)
                # local attention
                init_rt_mask = torch.repeat_interleave(X_init_rt, m_steps, axis=0).to(device)
                s_init_local, __ = local_featemb_init_attn(z_init, mask=init_rt_mask[:,:max_seg_len])
                
                # (bsize*k, mut_seqlen)
                m_shape = (X_mut_interp.shape[0], X_mut_interp.shape[1])
                x_mut_m =  mask_gen.create_content_mask(m_shape, x_mut_len_interp)
                max_seg_len = z_mut.shape[1]
                mut_mask = x_mut_m[:,:max_seg_len].to(device)
                s_mut_global, __ = global_featemb_mut_attn(z_mut, mask=mut_mask)
                mut_rt_mask = torch.repeat_interleave(X_mut_rt, m_steps, axis=0).to(device)
                s_mut_local, __ = local_featemb_mut_attn(z_mut, mask=mut_rt_mask[:,:max_seg_len])
                
                # print('s_init_global.shape:',s_init_global.shape)
                # print('s_init_local.shape:',s_init_local.shape)
                # print('s_mut_global.shape:', s_mut_global.shape)
                # print('s_mut_local.shape:', s_mut_local.shape)
                
                # (bsize*k, seqfeat_dim)
                seqlevel_interp = seqlevel_interp.reshape(-1, seqlevel_interp.shape[-1])
                z_seqfeat = seqlevel_featembeder(seqlevel_interp)
                # print('seqlevel_interp:', seqlevel_interp.shape)
                # print('z_seqfeat.shape:', z_seqfeat.shape)

                # y (bsize, num_outcomes)
                # (num_baselines*m_steps, num_outcomes)
                y_hat_logit = decoder(torch.cat([s_init_global, s_init_local, s_mut_global, s_mut_local, z_seqfeat], axis=-1))
                # print('y_hat_logit.shape:',y_hat_logit.shape)
                # print('y_batch.shape:',y_batch.shape)

                # TODO: try without the exponential part
                # (num_baselines*m_steps, 1)
                y_hat_logit = y_hat_logit[:,outcome_indx:outcome_indx+1]

                # list of gradients per input
                model_grads = torch.autograd.grad(outputs=y_hat_logit,
                                                  inputs=[X_init_interp, X_mut_interp, seqlevel_interp],
                                                  grad_outputs=torch.ones_like(y_hat_logit).to(device),
                                                  create_graph=True)

                # print(y_hat_logit.reshape(num_baselines, m_steps, -1))
                # y_logit shape: (num_baselines*m_steps, 1)
                y_hat_logit = y_hat_logit.detach()
                f_xinput = y_hat_logit.reshape(bsize, m_steps, -1)[:,-1,:]
                # f_xbase = y_hat_logit.reshape(bsize, m_steps, -1)[:,0:-1,:].mean(axis=1)
                f_xbase = y_hat_logit.reshape(bsize, m_steps, -1)[:,0,:]
                # print('f_xbase:', f_xbase)
                pred_score.extend(f_xinput[:,0].tolist())

                #TODO: add diagnositcs plots such as 
                # plot of alpha by average gradients 
                # add function to do sanity check on the computation of alpha
                
                # [num_baselines, m_steps, maxseqlen, embed_dim]

                # print()
                # print('init_grad.shape:',init_grad.shape)
                # print('mut_grad.shape:',mut_grad.shape)
                # print('seqfeat_grad.shape:',seqfeat_grad.shape)
                 

                ################
                # compute Integrated Gradients
                ###############
                # [num_baselines, k, max_seqlen, embedd_dim]
                init_grad = self._reshape_bsize_k(model_grads[0], bsize, m_steps)
                mut_grad = self._reshape_bsize_k(model_grads[1], bsize, m_steps)
                # [num_baselines, k, embedd_dim]
                seqfeat_grad = self._reshape_bsize_k(model_grads[2], bsize, m_steps)


                # TODO: test computing scaling first then computing the integral part!
                # [num_baselines, max_seqlen, embedd_dim]
                ig_init_grad_scaled = ig_explainer.compute_riemann_trapezoidal_approximation(init_grad*dist_init)
                ig_mut_grad_scaled = ig_explainer.compute_riemann_trapezoidal_approximation(mut_grad*dist_mut)
                # [num_baselines, seqlevelfeat_dim]
                ig_seqfeat_grad_scaled = ig_explainer.compute_riemann_trapezoidal_approximation(seqfeat_grad*dist_seqlevelfeat)
                
                # print('ig_init_grad_scaled.shape:',ig_init_grad_scaled.shape)
                # print('ig_mut_grad_scaled.shape:',ig_mut_grad_scaled.shape)
                # print('ig_seqfeat_grad_scaled.shape:',ig_seqfeat_grad_scaled.shape)
                
                
                
                #############
                # checking completness axiom by computing convergence diagnostic
                #############

                sum_attr_ = ig_explainer.sum_ig_contrib(ig_init_grad_scaled,x_init_len) + \
                            ig_explainer.sum_ig_contrib(ig_mut_grad_scaled, x_mut_len) + \
                            ig_seqfeat_grad_scaled.sum(axis=-1)

                
                # print('f_xinput.shape:', f_xinput.shape)
                # print('f_xinput', f_xinput)
                # print('f_xbase.shape:', f_xbase.shape)
                # print('f_xbase:', f_xbase)
                fsum_ = (f_xinput-f_xbase).squeeze(-1)
                # print('sum_attr ig_grad_scaled:', sum_attr_)
                # print('fsum:', fsum_)
                # diff = (sum_attr_-fsum_).abs()
                diff = (fsum_ - sum_attr_).abs()/fsum_.abs() # (delta_prediction - attribution_sum)/delta_prediction
                # print('diff abs:', diff)
                # print('diff relative (sum_attr_-fsum_).abs()/fsum_.abs()):', (sum_attr_-fsum_).abs()/fsum_.abs())
                # print('diff relative (fsum_- sum_attr_).abs()/fsum_.abs()):', (fsum_- sum_attr_).abs()/fsum_.abs())
                # print('diff relative (sum_attr_-fsum_).abs()/sum_attr.abs() :', (sum_attr_-fsum_).abs()/sum_attr_.abs())
                

                ##############
                # selecting top indices
                ##############
                
                convg_ = diff.tolist()

                # topk_indices = torch.topk(-1*diff,topk).indices
                # print('topk_indices:', topk_indices)
                # print(diff[topk_indices])
                # print(ig_init_grad_scaled[topk_indices].shape)
                # convg_ = (sum_attr_.mean()-fsum_.mean()).abs().item()
                # convg_ = diff[topk_indices].mean().item()
                # print('convg:', convg_)

                # sum_attr = ig_init_grad_scaled.abs().sum() + ig_mut_grad_scaled.abs().sum() + ig_seqfeat_grad_scaled.abs().sum()
                # fsum =  num_baselines*(f_xref-f_xbase).sum()
                # print('sum_attr (abs) ig_grad_scaled:', sum_attr.item())
                # print('fsum:', fsum.item())

                # sum_attr = ig_init_grad.abs().sum() + ig_mut_grad.abs().sum() + ig_seqfeat_grad.abs().sum()
                # fsum =  num_baselines*(f_xref-f_xbase).sum()
                # print('sum_attr (abs) ig_grad:', sum_attr.item())
                # print('fsum:', fsum.item())

                # (bsize, max_seqlen)
                ig_init_contrib = ig_init_grad_scaled.sum(axis=-1)
                # (bsize, max_seqlen)
                ig_mut_contrib = ig_mut_grad_scaled.sum(axis=-1)
                # (bsize, seqlevel_featuredim)
                ig_seqfeat_contrib = ig_seqfeat_grad_scaled

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


                convg_score_lst.append(convg_)
                # print('convg score: ', convg_score_lst)
                seqs_ids_lst.extend(b_seqs_id)
 
        return seqs_ids_lst, pred_score, ref_score, x_init_len_lst, x_mut_len_lst, ig_init_contrib_lst, ig_mut_contrib_lst, ig_seqfeat_contrib_lst, convg_score_lst

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

    def compute_integratedgradients_workflow(self, dloader, bg_set, model_dir, m_steps=50, y_ref=[]):

        mconfig_dir = os.path.join(model_dir, 'config')
        mconfig = self._load_model_config(mconfig_dir)
        
        # list of tuples (model, model_name)
        models = self._build_base_model(mconfig)

        models = self._load_model_statedict_(models, model_dir)
        # print(f'running prediction for prime editor | model_dir: {model_dir}')

        return self._get_integrated_gradients(models, 
                                             dloader,
                                             bg_set, 
                                             m_steps=m_steps, 
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