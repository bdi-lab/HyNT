from dataset import HNKG
from model import HyNT
from tqdm import tqdm
from utils import calculate_rank, metrics
import numpy as np
import argparse
import torch
import torch.nn as nn
import datetime
import time
import os
import copy
import math
import random

OMP_NUM_THREADS=8
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
torch.cuda.empty_cache()

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--data', default = "HN-WK", type = str)
parser.add_argument('--lr', default=4e-4, type=float)
parser.add_argument('--dim', default=256, type=int)
parser.add_argument('--num_epoch', default=1050, type=int)
parser.add_argument('--valid_epoch', default=150, type=int)
parser.add_argument('--exp', default='Reproduce')
parser.add_argument('--no_write', action='store_true')
parser.add_argument('--num_enc_layer', default=4, type=int)
parser.add_argument('--num_dec_layer', default=4, type=int)
parser.add_argument('--num_head', default=8, type=int)
parser.add_argument('--hidden_dim', default = 2048, type = int)
parser.add_argument('--dropout', default = 0.15, type = float)
parser.add_argument('--smoothing', default = 0.4, type = float)
parser.add_argument('--batch_size', default = 1024, type = int)
parser.add_argument('--step_size', default = 150, type = int)
parser.add_argument('--emb_as_proj', action = 'store_true')
args = parser.parse_args()


KG = HNKG(args.data, test = False)


KG_DataLoader = torch.utils.data.DataLoader(KG, batch_size = args.batch_size, shuffle=True)
model = HyNT(
	num_ent = KG.num_ent,
	num_rel = KG.num_rel,
    dim_model = args.dim,
    num_head = args.num_head,
    dim_hid = args.hidden_dim,
    num_enc_layer = args.num_enc_layer,
    num_dec_layer = args.num_dec_layer,
    dropout = args.dropout,
    emb_as_proj = args.emb_as_proj
).cuda()

criterion = nn.CrossEntropyLoss(label_smoothing = args.smoothing)
mse_criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.step_size, T_mult = 2)

file_format = f"{args.exp}/{args.data}/lr_{args.lr}_dim_{args.dim}_" + \
              f"elayer_{args.num_enc_layer}_dlayer_{args.num_dec_layer}_head_{args.num_head}_hid_{args.hidden_dim}_" + \
              f"drop_{args.dropout}_smoothing_{args.smoothing}_batch_{args.batch_size}_" + \
              f"steplr_{args.step_size}"

if args.emb_as_proj:
    file_format += "_embproj"

if not args.no_write:
    os.makedirs(f"./result/{args.exp}/{args.data}/", exist_ok=True)
    os.makedirs(f"./checkpoint/{args.exp}/{args.data}/", exist_ok=True)
    with open(f"./result/{file_format}.txt", "w") as f:
        f.write(f"{datetime.datetime.now()}\n")


start = time.time()
print("EPOCH \t TOTAL LOSS \t ENTITY LOSS \t RELATION LOSS \t NUMERIC LOSS \t TOTAL TIME")
for epoch in range(args.num_epoch):
    total_loss = 0.0
    total_ent_loss = 0.0
    total_rel_loss = 0.0
    total_num_loss = 0.0
    for batch, batch_pad, batch_mask_locs, answers, mask_idx, batch_num, ent_mask, rel_mask, num_mask, num_idx_mask, batch_real_len in KG_DataLoader:
        batch_len = max(batch_real_len)
        batch = batch[:,:batch_len]
        batch_pad = batch_pad[:,:batch_len//2]
        batch_mask_locs = batch_mask_locs[:,:batch_len//2]
        batch_num = batch_num[:,:batch_len//2]

        ent_score, rel_score, num_score = model(batch.cuda(), batch_num.cuda(), batch_pad.cuda(), batch_mask_locs.cuda())
        real_ent_mask = (ent_mask.cuda()!=0).squeeze()
        real_rel_mask = (rel_mask.cuda()!=0).squeeze()
        real_num_mask = (num_mask.cuda()!=0).squeeze()
        answer = answers.cuda()
        mask_idx = mask_idx.cuda()
        
        loss = 0
        if torch.any(ent_mask):
            real_ent_mask = real_ent_mask.cuda()
            ent_loss = criterion(ent_score[mask_idx][real_ent_mask], answer[real_ent_mask].long())
            loss += ent_loss
            total_ent_loss += ent_loss.item()

        if torch.any(rel_mask):
            real_rel_mask = real_rel_mask.cuda()
            rel_loss = criterion(rel_score[mask_idx][real_rel_mask], answer[real_rel_mask].long())
            loss += rel_loss
            total_rel_loss += rel_loss.item()

        if torch.any(num_mask):
            real_num_mask = real_num_mask.cuda()
            num_loss = mse_criterion(num_score[mask_idx][num_idx_mask], answer[real_num_mask])
            loss += num_loss
            total_num_loss += num_loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()

    print(f"{epoch} \t {total_loss:.6f} \t {total_ent_loss:.6f} \t" + \
          f"{total_rel_loss:.6f} \t {total_num_loss:.6f} \t {time.time() - start:.6f} s")

	### VALIDATION ###
    if (epoch + 1) % args.valid_epoch == 0:
        model.eval()

        lp_tri_list_rank = []
        lp_all_list_rank = []
        rp_tri_list_rank = []
        rp_all_list_rank = []
        nvp_tri_se = 0
        nvp_tri_se_num = 0
        nvp_all_se = 0
        nvp_all_se_num = 0
        with torch.no_grad():
            for tri, tri_pad, tri_num in tqdm(zip(KG.test, KG.test_pad, KG.test_num), total = len(KG.test)):

                tri_len = len(tri)
                pad_idx = 0
                for ent_idx in range((tri_len+1)//2):
                    if tri_pad[pad_idx]:
                        break
                    if ent_idx != 0:
                        pad_idx += 1
                    test_triplet = torch.tensor([tri])
                    
                    mask_locs = torch.full((1,(KG.max_len-3)//2+1), False)
                    if ent_idx < 2:
                        mask_locs[0,0] = True
                    else:
                        mask_locs[0,ent_idx-1] = True
                    if tri[ent_idx*2] >= KG.num_ent:
                        assert ent_idx != 0
                        test_num = torch.tensor([tri_num])
                        test_num[0,ent_idx-1] = -1
                        _,_,score_num = model(test_triplet.cuda(), test_num.cuda(), torch.tensor([tri_pad]).cuda(), mask_locs)
                        score_num = score_num.detach().cpu().numpy()
                        if ent_idx == 1:
                            sq_error = (score_num[0,3,tri[ent_idx*2]-KG.num_ent] - tri_num[ent_idx-1])**2
                            nvp_tri_se += sq_error
                            nvp_tri_se_num += 1
                        else:
                            sq_error = (score_num[0,2,tri[ent_idx*2]-KG.num_ent] - tri_num[ent_idx-1])**2
                        nvp_all_se += sq_error
                        nvp_all_se_num += 1
                    else:
                        test_triplet[0,2*ent_idx] = KG.num_ent+KG.num_rel
                        filt_tri = copy.deepcopy(tri)
                        filt_tri[ent_idx*2] = 2*(KG.num_ent+KG.num_rel)
                        if ent_idx != 1 and filt_tri[2] >= KG.num_ent:
                            re_pair = [(filt_tri[0], filt_tri[1], filt_tri[1] * 2 + tri_num[0])]
                        else:
                            re_pair = [(filt_tri[0], filt_tri[1], filt_tri[2])]
                        for qual_idx,(q,v) in enumerate(zip(filt_tri[3::2], filt_tri[4::2])):
                            if tri_pad[qual_idx+1]:
                                break
                            if ent_idx != qual_idx + 2 and v >= KG.num_ent:
                                re_pair.append((q, q*2 + tri_num[qual_idx + 1]))
                            else:
                                re_pair.append((q,v))
                        re_pair.sort()
                        filt = KG.filter_dict[tuple(re_pair)]
                        score_ent, _, _ = model(test_triplet.cuda(), torch.tensor([tri_num]).cuda(), torch.tensor([tri_pad]).cuda(), mask_locs)
                        score_ent = score_ent.detach().cpu().numpy()
                        if ent_idx < 2:
                            rank = calculate_rank(score_ent[0,1+2*ent_idx],tri[ent_idx*2], filt)
                            lp_tri_list_rank.append(rank)
                        else:
                            rank = calculate_rank(score_ent[0,2], tri[ent_idx*2], filt)
                        lp_all_list_rank.append(rank)
                for rel_idx in range(tri_len//2):
                    if tri_pad[rel_idx]:
                        break
                    mask_locs = torch.full((1,(KG.max_len-3)//2+1), False)
                    mask_locs[0,rel_idx] = True
                    test_triplet = torch.tensor([tri])
                    orig_rels = tri[1::2]
                    test_triplet[0, rel_idx*2 + 1] = KG.num_rel
                    if test_triplet[0, rel_idx*2+2] >= KG.num_ent:
                        test_triplet[0, rel_idx*2 + 2] = KG.num_ent + KG.num_rel
                    filt_tri = copy.deepcopy(tri)
                    filt_tri[rel_idx*2+1] = 2*(KG.num_ent+KG.num_rel)
                    if filt_tri[2] >= KG.num_ent:
                        re_pair = [(filt_tri[0], filt_tri[1], orig_rels[0]*2 + tri_num[0])]
                    else:
                        re_pair = [(filt_tri[0], filt_tri[1], filt_tri[2])]
                    for qual_idx,(q,v) in enumerate(zip(filt_tri[3::2], filt_tri[4::2])):
                        if tri_pad[qual_idx+1]:
                            break
                        if v >= KG.num_ent:
                            re_pair.append((q, orig_rels[qual_idx + 1]*2 + tri_num[qual_idx + 1]))
                        else:
                            re_pair.append((q,v))
                    re_pair.sort()
                    filt = KG.filter_dict[tuple(re_pair)]
                    _,score_rel, _ = model(test_triplet.cuda(), torch.tensor([tri_num]).cuda(), torch.tensor([tri_pad]).cuda(), mask_locs)
                    score_rel = score_rel.detach().cpu().numpy()
                    if rel_idx == 0:
                        rank = calculate_rank(score_rel[0,2], tri[rel_idx*2+1], filt)
                        rp_tri_list_rank.append(rank)
                    else:
                        rank = calculate_rank(score_rel[0,1], tri[rel_idx*2+1], filt)
                    rp_all_list_rank.append(rank)
                    
        lp_tri_list_rank = np.array(lp_tri_list_rank)
        lp_tri_mrr, lp_tri_hit10, lp_tri_hit3, lp_tri_hit1 = metrics(lp_tri_list_rank)
        print("Link Prediction on Validation Set (Tri)")
        print(f"MRR: {lp_tri_mrr:.4f}")
        print(f"Hit@10: {lp_tri_hit10:.4f}")
        print(f"Hit@3: {lp_tri_hit3:.4f}")
        print(f"Hit@1: {lp_tri_hit1:.4f}")

        lp_all_list_rank = np.array(lp_all_list_rank)
        lp_all_mrr, lp_all_hit10, lp_all_hit3, lp_all_hit1 = metrics(lp_all_list_rank)
        print("Link Prediction on Validation Set (All)")
        print(f"MRR: {lp_all_mrr:.4f}")
        print(f"Hit@10: {lp_all_hit10:.4f}")
        print(f"Hit@3: {lp_all_hit3:.4f}")
        print(f"Hit@1: {lp_all_hit1:.4f}")
    
        rp_tri_list_rank = np.array(rp_tri_list_rank)
        rp_tri_mrr, rp_tri_hit10, rp_tri_hit3, rp_tri_hit1 = metrics(rp_tri_list_rank)
        print("Relation Prediction on Validation Set (Tri)")
        print(f"MRR: {rp_tri_mrr:.4f}")
        print(f"Hit@10: {rp_tri_hit10:.4f}")
        print(f"Hit@3: {rp_tri_hit3:.4f}")
        print(f"Hit@1: {rp_tri_hit1:.4f}")

        rp_all_list_rank = np.array(rp_all_list_rank)
        rp_all_mrr, rp_all_hit10, rp_all_hit3, rp_all_hit1 = metrics(rp_all_list_rank)
        print("Relation Prediction on Validation Set (All)")
        print(f"MRR: {rp_all_mrr:.4f}")
        print(f"Hit@10: {rp_all_hit10:.4f}")
        print(f"Hit@3: {rp_all_hit3:.4f}")
        print(f"Hit@1: {rp_all_hit1:.4f}")

        if nvp_tri_se_num > 0:
            nvp_tri_rmse = math.sqrt(nvp_tri_se/nvp_tri_se_num)
            print("Numeric Value Prediction on Validation Set (Tri)")
            print(f"RMSE: {nvp_tri_rmse:.4f}")

        if nvp_all_se_num > 0:
            nvp_all_rmse = math.sqrt(nvp_all_se/nvp_all_se_num)
            print("Numeric Value Prediction on Validation Set (All)")
            print(f"RMSE: {nvp_all_rmse:.4f}")

        if not args.no_write:
            with open(f"./result/{file_format}.txt", 'a') as f:
                f.write(f"Epoch: {epoch+1}\n")
                f.write(f"Link Prediction on Validation Set (Tri): {lp_tri_mrr:.4f} {lp_tri_hit10:.4f} {lp_tri_hit3:.4f} {lp_tri_hit1:.4f}\n")
                f.write(f"Link Prediction on Validation Set (All): {lp_all_mrr:.4f} {lp_all_hit10:.4f} {lp_all_hit3:.4f} {lp_all_hit1:.4f}\n")
                f.write(f"Relation Prediction on Validation Set (Tri): {rp_tri_mrr:.4f} {rp_tri_hit10:.4f} {rp_tri_hit3:.4f} {rp_tri_hit1:.4f}\n")
                f.write(f"Relation Prediction on Validation Set (All): {rp_all_mrr:.4f} {rp_all_hit10:.4f} {rp_all_hit3:.4f} {rp_all_hit1:.4f}\n")
                if nvp_tri_se_num > 0:
                    f.write(f"Numeric Value Prediction on Validation Set (Tri): {nvp_tri_rmse:.4f}\n")
                if nvp_all_se_num > 0:
                    f.write(f"Numeric Value Prediction on Validation Set (All): {nvp_all_rmse:.4f}\n")
                    
            
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                        f"./checkpoint/{file_format}_{epoch+1}.ckpt")

        model.train()
