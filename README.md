# Representation Learning on Hyper-Relational and Numeric Knowledge Graphs with Transformers
This code is the official implementation of the following [paper](https://arxiv.org/abs/2305.18256):

> Chanyoung Chung, Jaejun Lee, and Joyce Jiyoung Whang, Representation Learning on Hyper-Relational and Numeric Knowledge Graphs with Transformers, The 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2023.

All codes are written by Jaejun Lee (jjlee98@kaist.ac.kr). When you use this code or data, please cite our paper.
```bibtex
@inproceedings{hynt,
	author={Chanyoung Chung and Jaejun Lee and Joyce Jiyoung Whang},
	title={Representation Learning on Hyper-Relational and Numeric Knowledge Graphs with Transformers},
	booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
	year={2023},
	pages={310--322},
	doi={10.1145/3580305.3599490}
}
```

## Updates on Jan. 5th, 2024

We have re-uploaded our codes after fixing a bug in filter_dict. 

## Updates on July. 10th, 2024

The link prediction results on HN-KGs are updated, while the other experimental results remain the same. You can find the updated results in https://arxiv.org/abs/2305.18256.

## Updates on Oct. 11th, 2024

We have added a checkpoint file for WD50K by training HyNT using both the training and validation sets.
The experimental results can be found on [arXiv](https://arxiv.org/abs/2305.18256).

## Requirements

We used python 3.7 and PyTorch 1.12.0 with cudatoolkit 11.3.

You can install all requirements with:

```shell
pip install -r requirements.txt
```

## Reproducing the Reported Results

We used NVIDIA RTX A6000, NVIDIA GeForce RTX 3090 or NVIDIA GeForce RTX 2080Ti for all our experiments. We provide the checkpoints to produce the link prediction, relation prediction, and numeric value prediction results on HN-WK, HN-YG, HN-FB, and HN-FB-S. The checkpoints are also provided for the link prediction results on WD50K and WikiPeople<sup>$\mathbf{-}$</sup>. If you want to use the checkpoints, place the unzipped checkpoint folder in the same directory with the codes.

You can download the checkpoints from https://drive.google.com/file/d/1CY6S6iBm63Bp3Fl3HjB9s4DSNVEkB-FG/view?usp=sharing.

The commands to reproduce the results in our paper:

### HN-WK

#### Link Prediction

```python
python3 eval.py --data HN-WK --lr 4e-4 --dim 256 --epoch 1050 --exp KDD --num_enc_layer 4 --num_dec_layer 4 --num_head 8 --hidden_dim 2048 --dropout 0.15 --smoothing 0.4 --batch_size 1024 --step_size 150 --lp
```

#### Relation Prediction

```python
python3 eval.py --data HN-WK --lr 4e-4 --dim 256 --epoch 1050 --exp KDD --num_enc_layer 4 --num_dec_layer 4 --num_head 8 --hidden_dim 2048 --dropout 0.15 --smoothing 0.4 --batch_size 1024 --step_size 150 --rp
```

#### Numeric Value Prediction

```python
python3 eval.py --data HN-WK --lr 1e-3 --dim 256 --epoch 750 --exp KDD --num_enc_layer 2 --num_dec_layer 2 --num_head 16 --hidden_dim 1024 --dropout 0.1 --smoothing 0.5 --batch_size 1024 --step_size 50 --nvp
```

### HN-YG

#### Link Prediction

```python
python3 eval.py --data HN-YG --lr 1e-3 --dim 256 --epoch 700 --exp KDD --num_enc_layer 2 --num_dec_layer 2 --num_head 64 --hidden_dim 1536 --dropout 0.4 --smoothing 0.7 --batch_size 2048 --step_size 50 --lp
```

#### Relation Prediction

```python
python3 eval.py --data HN-YG --lr 1e-3 --dim 256 --epoch 50 --exp KDD --num_enc_layer 2 --num_dec_layer 2 --num_head 16 --hidden_dim 1024 --dropout 0.1 --smoothing 0.5 --batch_size 2048 --step_size 50 --rp
```

#### Numeric Value Prediction

```python
python3 eval.py --data HN-YG --lr 1e-3 --dim 256 --epoch 350 --exp KDD --num_enc_layer 2 --num_dec_layer 2 --num_head 16 --hidden_dim 1024 --dropout 0.1 --smoothing 0.5 --batch_size 2048 --step_size 50 --nvp
```

### HN-FB

#### Link Prediction

```python
python3 eval.py --data HN-FB --lr 1e-3 --dim 256 --epoch 750 --exp KDD --num_enc_layer 2 --num_dec_layer 2 --num_head 16 --hidden_dim 1024 --dropout 0.1 --smoothing 0.3 --batch_size 512 --step_size 50 --lp
```

#### Relation Prediction

```python
python3 eval.py --data HN-FB --lr 1e-3 --dim 256 --epoch 50 --exp KDD --num_enc_layer 2 --num_dec_layer 2 --num_head 16 --hidden_dim 1024 --dropout 0.1 --smoothing 0.3 --batch_size 512 --step_size 50 --rp
```

#### Numeric Value Prediction

```python
python3 eval.py --data HN-FB --lr 1e-3 --dim 256 --epoch 750 --exp KDD --num_enc_layer 2 --num_dec_layer 2 --num_head 16 --hidden_dim 1024 --dropout 0.1 --smoothing 0.3 --batch_size 512 --step_size 50 --nvp
```

### HN-FB-S

#### Link Prediction

```python
python3 eval.py --data HN-FB-S --lr 3e-4 --dim 256 --epoch 1550 --exp KDD --num_enc_layer 10 --num_dec_layer 3 --num_head 8 --hidden_dim 1024 --dropout 0.3 --smoothing 0.55 --batch_size 512 --step_size 50 --emb_as_proj --lp
```

#### Relation Prediction

```python
python3 eval.py --data HN-FB-S --lr 3e-4 --dim 256 --epoch 600 --exp KDD --num_enc_layer 6 --num_dec_layer 3 --num_head 4 --hidden_dim 2048 --dropout 0.35 --smoothing 0.45 --batch_size 512 --step_size 50 --rp
```

#### Numeric Value Prediction

```python
python3 eval.py --data HN-FB-S --lr 1e-3 --dim 256 --epoch 750 --exp KDD --num_enc_layer 2 --num_dec_layer 2 --num_head 16 --hidden_dim 1024 --dropout 0.2 --smoothing 0.7 --batch_size 2048 --step_size 50 --nvp
```


### WikiPeople-

#### Link Prediction

```python
python3 eval.py --data WikiPeople- --lr 1e-3 --dim 256 --epoch 350 --exp KDD --num_enc_layer 3 --num_dec_layer 3 --num_head 16 --hidden_dim 1024 --dropout 0.2 --smoothing 0.4 --batch_size 2048 --step_size 50 --lp
```

### WD50K

#### Link Prediction

```python
python3 eval.py --data WD50K --lr 1e-3 --dim 256 --epoch 350 --exp KDD --num_enc_layer 3 --num_dec_layer 3 --num_head 4 --hidden_dim 1024 --dropout 0.2 --smoothing 0.7 --batch_size 2048 --step_size 50 --lp
```

### WD50K-test

#### Link Prediction

```python
python3 eval.py --data WD50K-test --lr 1e-3 --dim 256 --epoch 350 --exp KDD --num_enc_layer 3 --num_dec_layer 3 --num_head 4 --hidden_dim 1024 --dropout 0.2 --smoothing 0.7 --batch_size 2048 --step_size 50 --lp
```

## Training from Scratch

To train HyNT from scratch, run `train.py` with arguments. Please refer to `train.py` or `eval.py` for the examples of the arguments.

The list of arguments of 'train.py':
- `--data`: name of the dataset
- `--lr`: learning rate
- `--dim`: $d=\hat{d}$
- `--num_epoch`: total number of training epochs (only used for train.py)
- `--epoch`: the epoch to evaluate (only used for eval.py)
- `--valid_epoch`: the duration of validation
- `--exp`: experiment name
- `--num_enc_layer`: $L_\mathrm{C}$
- `--num_dec_layer`: $L_\mathrm{P}$
- `--num_head`: $n_\mathrm{P}=n_\mathrm{C}$
- `--hidden_dim`: $d_\mathrm{F}=\hat{d}_\mathrm{F}$
- `--dropout`: $\delta$
- `--smoothing`: $\epsilon$
- `--batch_size`: the batch size
- `--step_size`: the step size of the cosine annealing learning rate scheduler

### Hyperparameters
We tuned HyNT on each dataset with the following tuning range:

#### HN-WK
- lr: {0.0003, 0.0004, 0.0005, 0.001}
- dim: 256
- num_epoch: 750 if step_size is 50, 1050 if step_size is 150
- valid_epoch: equal to step_size
- num_enc_layer: {2, 3, 4}
- num_dec_layer: {2, 3, 4}
- num_head: {8, 16}
- hidden_dim: {1024, 2048}
- dropout: {0.1, 0.15, 0.2}
- smoothing: {0.3, 0.4, 0.5}
- batch_size: {1024, 2048}
- step_size: {50, 150}

#### HN-YG
- lr: 0.001
- dim: 256
- num_epoch: 750 if step_size is 50, 700 if step_size is 100
- valid_epoch: 50
- num_enc_layer: 2
- num_dec_layer: 2
- num_head: {16, 32, 64}
- hidden_dim: {1024, 1536, 2048}
- dropout: {0.1, 0.2, 0.3, 0.4, 0.5}
- smoothing: {0.3, 0.5, 0.7}
- batch_size: {2048}
- step_size: {50, 100}

#### HN-FB-S
- lr: {0.0003, 0.0005, 0.001}
- dim: 256
- num_epoch: 1550
- valid_epoch: 50
- num_enc_layer: {2, 4, 6, 8, 10}
- num_dec_layer: {2, 3, 4}
- num_head: {4, 8, 16}
- hidden_dim: {1024, 2048}
- dropout: {0.2, 0.25, 0.3, 0.35}
- smoothing: {0.45, 0.5, 0.55, 0.6, 0.7}
- batch_size: {512, 1024, 2048}
- step_size: 50
- emb_as_proj: {on, off}

#### HN-FB
- lr: 0.001
- dim: 256
- num_epoch: 750 if step_size is 50, 700 if step_size is 100
- valid_epoch: 50
- num_enc_layer: 2
- num_dec_layer: 2
- num_head: {4, 8, 16}
- hidden_dim: 1024
- dropout: {0.1, 0.2}
- smoothing: {0.3, 0.5, 0.7}
- batch_size: 512
- step_size: {50, 100}

#### WikiPeople-
- lr: 0.001
- dim: 256
- num_epoch: 750 if step_size is 50, 700 if step_size is 100
- valid_epoch: 50
- num_enc_layer: {2, 3}
- num_dec_layer: {2, 3}
- num_head: {4, 8, 16}
- hidden_dim: 1024
- dropout: {0.1, 0.2}
- smoothing: {0.3, 0.4, 0.5}
- batch_size: 2048
- step_size: {50, 100}

#### WD50K
- lr: 0.001
- dim: 256
- num_epoch: 750 if step_size is 50, 700 if step_size is 100
- valid_epoch: 50
- num_enc_layer: {2, 3}
- num_dec_layer: {2, 3}
- num_head: {4, 8, 16}
- hidden_dim: 1024
- dropout: {0.1, 0.2}
- smoothing: {0.3, 0.5, 0.7}
- batch_size: 2048
- step_size: {50, 100}

## Generating Datasets

You can download the codes for generating datasets from https://drive.google.com/file/d/1NOThGP_wXdQeGgEGut3Pzn-Jth96jIIs/view?usp=sharing.

## License
Our datasets and codes are released under the CC BY-NC-SA 4.0 license.
