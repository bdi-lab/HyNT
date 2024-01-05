import torch
from torch.utils.data import Dataset
import numpy as np
import copy

class HNKG(Dataset):
    def __init__(self, data, test = False):
        self.data = data
        self.dir = "data_normalized/{}/".format(self.data)

        self.ent2id = {}
        self.id2ent = {}
        with open(self.dir+"entity2id.txt") as f:
            lines = f.readlines()
            self.num_ent = int(lines[0].strip())
            for line in lines[1:]:
                ent, idx = line.strip().split("\t")
                self.ent2id[ent] = int(idx)
                self.id2ent[int(idx)] = ent

        self.rel2id = {}
        self.id2rel = {}
        with open(self.dir+"relation2id.txt") as f:
            lines = f.readlines()
            self.num_rel = int(lines[0].strip())
            for line in lines[1:]:
                rel, idx = line.strip().split("\t")
                self.rel2id[rel] = int(idx)
                self.id2rel[int(idx)] = rel

        self.train = []
        self.train_pad = []
        self.train_num = []
        self.train_len = []
        self.max_len = 0
        with open(self.dir+"train.txt") as f:
            for line in f.readlines()[1:]:
                hp_triplet = line.strip().split("\t")
                h,r,t = hp_triplet[:3]
                num_qual = (len(hp_triplet)-3)//2
                self.train_len.append(len(hp_triplet))
                try:
                    self.train_num.append([float(t)])
                    self.train.append([self.ent2id[h],self.rel2id[r],self.num_ent+self.rel2id[r]])
                except:
                    self.train.append([self.ent2id[h],self.rel2id[r],self.ent2id[t]])
                    self.train_num.append([1])
                self.train_pad.append([False])
                for i in range(num_qual):
                    q = hp_triplet[3+2*i]
                    v = hp_triplet[4+2*i]
                    self.train[-1].append(self.rel2id[q])
                    try:
                        self.train_num[-1].append(float(v))
                        self.train[-1].append(self.num_ent+self.rel2id[q])
                    except:
                        self.train_num[-1].append(1)
                        self.train[-1].append(self.ent2id[v])
                    self.train_pad[-1].append(False)
                tri_len = num_qual*2+3
                if tri_len > self.max_len:
                    self.max_len = tri_len
        self.num_train = len(self.train)
        for i in range(self.num_train):
            curr_len = len(self.train[i])
            for j in range((self.max_len-curr_len)//2):
                self.train[i].append(0)
                self.train[i].append(0)
                self.train_pad[i].append(True)
                self.train_num[i].append(1)

        self.test = []
        self.test_pad = []
        self.test_num = []
        self.test_len = []
        if test:
            test_dir = self.dir + "test.txt"
        else:
            test_dir = self.dir + "valid.txt"
        with open(test_dir) as f:
            for line in f.readlines()[1:]:
                hp_triplet = []
                hp_pad = []
                hp_num = []
                for i, anything in enumerate(line.strip().split("\t")):
                    if i % 2 == 0 and i != 0:
                        try:
                            hp_num.append(float(anything))
                            hp_triplet.append(self.num_ent + hp_triplet[-1])
                        except:
                            hp_triplet.append(self.ent2id[anything])
                            hp_num.append(1)
                    elif i == 0:
                        hp_triplet.append(self.ent2id[anything])
                    else:
                        hp_triplet.append(self.rel2id[anything])
                        hp_pad.append(False)
                flag = 0
                self.test_len.append(len(hp_triplet))
                while len(hp_triplet) < self.max_len:
                    hp_triplet.append(0)
                    flag += 1
                    if flag % 2:
                        hp_num.append(1)
                        hp_pad.append(True)
                self.test.append(hp_triplet)
                self.test_pad.append(hp_pad)
                self.test_num.append(hp_num)

        self.num_test = len(self.test)

        self.valid = []
        self.valid_pad = []
        self.valid_num = []
        self.valid_len = []
        if test:
            valid_dir = self.dir + "valid.txt"
        else:
            valid_dir = self.dir + "test.txt"
        with open(valid_dir) as f:
            for line in f.readlines()[1:]:
                hp_triplet = []
                hp_pad = []
                hp_num = []
                for i, anything in enumerate(line.strip().split("\t")):
                    if i % 2 == 0 and i != 0:
                        try:
                            hp_num.append(float(anything))
                            hp_triplet.append(self.num_ent + hp_triplet[-1])
                        except:
                            hp_triplet.append(self.ent2id[anything])
                            hp_num.append(1)
                    elif i == 0:
                        hp_triplet.append(self.ent2id[anything])
                    else:
                        hp_triplet.append(self.rel2id[anything])
                        hp_pad.append(False)
                flag = 0
                self.valid_len.append(len(hp_triplet))
                while len(hp_triplet) < self.max_len:
                    hp_triplet.append(0)
                    flag += 1
                    if flag % 2:
                        hp_num.append(1)
                        hp_pad.append(True)
                self.valid.append(hp_triplet)
                self.valid_pad.append(hp_pad)
                self.valid_num.append(hp_num)
        self.num_valid = len(self.valid)

        self.filter_dict = self.construct_filter_dict()
        self.train = torch.tensor(self.train)
        self.train_pad = torch.tensor(self.train_pad)
        self.train_num = torch.tensor(self.train_num)
        self.train_len = torch.tensor(self.train_len)

    def __len__(self):
        return self.num_train

    def __getitem__(self, idx):
        masked = self.train[idx].clone()
        masked_num = self.train_num[idx].clone()
        mask_idx = np.random.randint(self.train_len[idx])

        if mask_idx % 2 == 0:
            if self.train[idx, mask_idx] < self.num_ent:
                masked[mask_idx] = self.num_ent+self.num_rel
        else:
            masked[mask_idx] = self.num_rel
            if masked[mask_idx+1] >= self.num_ent:
                masked[mask_idx+1] = self.num_ent+self.num_rel
        answer = self.train[idx, mask_idx]

        mask_locs = torch.full(((self.max_len-3)//2+1,), False)
        if mask_idx < 3:
            mask_locs[0] = True
        else:
            mask_locs[(mask_idx-3)//2+1] = True
        
        mask_idx_mask = torch.full((4,), False)
        if mask_idx < 3:
            mask_idx_mask[mask_idx+1] = True
        else:
            mask_idx_mask[2-mask_idx%2] = True
        
        num_idx_mask = torch.full((self.num_rel,),False)
        if mask_idx % 2 == 0:
            if self.train[idx, mask_idx] >= self.num_ent:
                num_idx_mask[self.train[idx,mask_idx]-self.num_ent] = True
                answer = self.train_num[idx, (mask_idx-1)//2]
                masked_num[mask_idx//2-1] = -1
                ent_mask = [0]
                num_mask = [1]
            else:
                num_mask = [0]
                ent_mask = [1]
            rel_mask = [0]
        else:
            num_mask = [0]
            ent_mask = [0]
            rel_mask = [1]
        
        return masked, self.train_pad[idx], mask_locs, answer, mask_idx_mask, masked_num, torch.tensor(ent_mask), torch.tensor(rel_mask), torch.tensor(num_mask), num_idx_mask, self.train_len[idx]

    def max_len(self):
        return self.max_len

    def construct_filter_dict(self):
        res = {}
        for data, data_len, data_num in [[self.train, self.train_len, self.train_num],[self.valid, self.valid_len, self.valid_num],[self.test, self.test_len, self.test_num]]:
            for triplet, triplet_len, triplet_num in zip(data, data_len, data_num):
                real_triplet = copy.deepcopy(triplet[:triplet_len])
                if real_triplet[2] < self.num_ent:
                    re_pair = [(real_triplet[0], real_triplet[1], real_triplet[2])]
                else:
                    re_pair = [(real_triplet[0], real_triplet[1], real_triplet[1]*2 + triplet_num[0])]
                for idx, (q,v) in enumerate(zip(real_triplet[3::2], real_triplet[4::2])):
                    if v <self.num_ent:
                        re_pair.append((q, v))
                    else:
                        re_pair.append((q, q*2 + triplet_num[idx + 1]))
                for i, pair in enumerate(re_pair):
                    for j, anything in enumerate(pair):
                        filtered_filter = copy.deepcopy(re_pair)
                        new_pair = copy.deepcopy(list(pair))
                        new_pair[j] = 2*(self.num_ent+self.num_rel)
                        filtered_filter[i] = tuple(new_pair)
                        filtered_filter.sort()
                        try:
                            res[tuple(filtered_filter)].append(pair[j])
                        except:
                            res[tuple(filtered_filter)] = [pair[j]]
        for key in res:
            res[key] = np.array(res[key])

        return res

