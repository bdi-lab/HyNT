import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class HyNT(nn.Module):
    def __init__(self, num_ent, num_rel, dim_model, num_head, dim_hid, num_enc_layer, num_dec_layer, dropout = 0.1, emb_as_proj = False):
        super(HyNT, self).__init__()
        self.dim_model = dim_model
        self.num_head = num_head
        self.dim_hid = dim_hid
        self.num_enc_layer = num_enc_layer
        self.num_dec_layer = num_dec_layer
        self.dropout = dropout
        self.pri_pos = nn.Parameter(torch.Tensor(1, 1, dim_model))
        self.qv_pos = nn.Parameter(torch.Tensor(1, 1, dim_model))
        self.h_pos = nn.Parameter(torch.Tensor(1, 1, dim_model))
        self.r_pos = nn.Parameter(torch.Tensor(1, 1, dim_model))
        self.t_pos = nn.Parameter(torch.Tensor(1, 1, dim_model))
        self.q_pos = nn.Parameter(torch.Tensor(1, 1, dim_model))
        self.v_pos = nn.Parameter(torch.Tensor(1, 1, dim_model))

        self.ent_embeddings = nn.Embedding(num_ent+1+num_rel, dim_model)
        self.rel_embeddings = nn.Embedding(num_rel+1, dim_model)
        self.pri_enc = nn.Linear(dim_model*3, dim_model)
        self.qv_enc = nn.Linear(dim_model*2, dim_model)

        self.ent_dec = nn.Linear(dim_model, num_ent)
        self.rel_dec = nn.Linear(dim_model, num_rel)
        self.num_dec = nn.Linear(dim_model, num_rel)

        self.num_mask = nn.Parameter(torch.tensor(0.5))

        encoder_layer = nn.TransformerEncoderLayer(dim_model, num_head, dim_hid, dropout, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_enc_layer)
        decoder_layer = nn.TransformerEncoderLayer(dim_model, num_head, dim_hid, dropout, batch_first = True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_dec_layer)

        self.emb_as_proj = emb_as_proj
        self.num_ent = num_ent

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.pri_pos)
        nn.init.xavier_uniform_(self.qv_pos)
        nn.init.xavier_uniform_(self.h_pos)
        nn.init.xavier_uniform_(self.r_pos)
        nn.init.xavier_uniform_(self.t_pos)
        nn.init.xavier_uniform_(self.q_pos)
        nn.init.xavier_uniform_(self.v_pos)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.pri_enc.weight)
        nn.init.xavier_uniform_(self.qv_enc.weight)
        nn.init.xavier_uniform_(self.ent_dec.weight)
        nn.init.xavier_uniform_(self.rel_dec.weight)
        nn.init.xavier_uniform_(self.num_dec.weight)
        self.pri_enc.bias.data.zero_()
        self.qv_enc.bias.data.zero_()
        self.ent_dec.bias.data.zero_()
        self.rel_dec.bias.data.zero_()
        self.num_dec.bias.data.zero_()

    def forward(self, src, num_values, src_key_padding_mask, mask_locs):
        batch_size = len(src)
        num_val = torch.where(num_values != -1, num_values, self.num_mask)
        h_seq = self.ent_embeddings(src[...,0]).view(batch_size, 1, self.dim_model)
        r_seq = self.rel_embeddings(src[...,1]).view(batch_size, 1, self.dim_model)
        t_seq = (self.ent_embeddings(src[...,2])*num_val[...,0:1]).view(batch_size, 1, self.dim_model)
        q_seq = self.rel_embeddings(src[...,3::2].flatten()).view(batch_size, -1, self.dim_model)
        v_seq = (self.ent_embeddings(src[...,4::2].flatten())*num_val[...,1:].flatten().unsqueeze(-1)).view(batch_size, -1, self.dim_model)

        tri_seq = self.pri_enc(torch.cat([h_seq, r_seq, t_seq], dim = -1)) + self.pri_pos
        qv_seqs = self.qv_enc(torch.cat([q_seq, v_seq], dim= -1)) + self.qv_pos

        enc_in_seq = torch.cat([tri_seq, qv_seqs], dim = 1)
        enc_out_seq = self.encoder(enc_in_seq, src_key_padding_mask = src_key_padding_mask)

        dec_in_rep = enc_out_seq[mask_locs].view(batch_size, 1, self.dim_model)
        triplet = torch.stack([h_seq + self.h_pos, r_seq + self.r_pos, t_seq + self.t_pos], dim = 2)
        qv = torch.stack([q_seq + self.q_pos, v_seq + self.v_pos, torch.zeros_like(v_seq)], dim = 2)
        dec_in_part = torch.cat([triplet,qv], dim = 1)[mask_locs]
        dec_in_seq = torch.cat([dec_in_rep, dec_in_part], dim = 1)
        dec_in_mask = torch.full((batch_size,4),False).cuda()
        dec_in_mask[torch.nonzero(mask_locs==1)[:,1]!=0,3] = True
        dec_out_seq = self.decoder(dec_in_seq, src_key_padding_mask = dec_in_mask)
        
        if self.emb_as_proj:
            ent_out = torch.matmul(dec_out_seq, self.ent_embeddings.weight[:self.num_ent].T) + self.ent_dec.bias
        else:
            ent_out = self.ent_dec(dec_out_seq)
        return ent_out, self.rel_dec(dec_out_seq), self.num_dec(dec_out_seq)
