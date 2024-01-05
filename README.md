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

## Updates

We have re-uploaded our codes after fixing a bug in filter_dict. The link prediction results on HN-KGs need to be updated, while the other experimental results remain the same. Tentatively, we report the updated results on HN-WK below with limited parameter tuning. We will release the full updated results and the checkpoints as soon as possible.

Link Prediction on the Primary Triplets in HN-WK (Updated)\
MRR: 0.3025 / Hit10: 0.5037 / Hit3: 0.3214 / Hit1: 0.2075

Link Prediction on All Entities of the Hyper-Relational Facts in HN-WK (Updated)\
MRR: 0.3234 / Hit10: 0.5216 / Hit3: 0.3439 / Hit1: 0.2294

## Requirements

We used python 3.7 and PyTorch 1.12.0 with cudatoolkit 11.3.

You can install all requirements with:

```shell
pip install -r requirements.txt
```

## Reproducing the Reported Results

We used NVIDIA RTX A6000 and NVIDIA GeForce RTX 3090 for all our experiments. We provide the checkpoints to produce the link prediction, relation prediction, and numeric value prediction results on HN-WK, HN-YG, HN-FB, and HN-FB-S. The checkpoints are also provided for the link prediction results on WD50K and WikiPeople<sup>$\mathbf{-}$</sup>. If you want to use the checkpoints, place the unzipped checkpoint folder in the same directory with the codes.

You can download the checkpoints from https://drive.google.com/file/d/1EUg7n5vsfnrT-R0B6851y7RJvjeWYTyo/view?usp=sharing.

The commands to reproduce the results in our paper:

### HN-WK

#### Link Prediction & Numeric Value Prediction

```python
python3 eval.py --data HN-WK --lr 1e-3 --dim 256 --epoch 750 --exp lp_nvp --num_layer 2 --num_head 16 --hidden_dim 1024 --dropout 0.1 --smoothing 0.5 --batch_size 1024 --step_size 50 --lp --nvp
```

#### Relation Prediction

```python
python3 eval.py --data HN-WK --lr 1e-3 --dim 256 --epoch 750 --exp rp --num_layer 2 --num_head 16 --hidden_dim 1024 --dropout 0.1 --smoothing 0.5 --batch_size 1024 --step_size 50 --rp
```

### HN-YG

#### Link Prediction & Numeric Value Prediction

```python
python3 eval.py --data HN-YG --lr 1e-3 --dim 256 --epoch 350 --exp lp_nvp --num_layer 2 --num_head 16 --hidden_dim 1024 --dropout 0.1 --smoothing 0.5 --batch_size 2048 --step_size 50 --lp --nvp
```

#### Relation Prediction

```python
python3 eval.py --data HN-YG --lr 1e-3 --dim 256 --epoch 50 --exp rp --num_layer 2 --num_head 16 --hidden_dim 1024 --dropout 0.1 --smoothing 0.5 --batch_size 2048 --step_size 50 --rp
```

### HN-FB

#### Link Prediction & Numeric Value Prediction

```python
python3 eval.py --data HN-FB --lr 1e-3 --dim 256 --epoch 750 --exp lp_nvp --num_layer 2 --num_head 16 --hidden_dim 1024 --dropout 0.1 --smoothing 0.3 --batch_size 512 --step_size 50 --lp --nvp
```

#### Relation Prediction

```python
python3 eval.py --data HN-FB --lr 1e-3 --dim 256 --epoch 50 --exp rp --num_layer 2 --num_head 16 --hidden_dim 1024 --dropout 0.1 --smoothing 0.3 --batch_size 512 --step_size 50 --rp
```

### HN-FB-S

#### Link Prediction & Numeric Value Prediction

```python
python3 eval.py --data HN-FB-S --lr 1e-3 --dim 256 --epoch 750 --exp lp_nvp --num_layer 2 --num_head 16 --hidden_dim 1024 --dropout 0.2 --smoothing 0.7 --batch_size 2048 --step_size 50 --lp --nvp
```

#### Relation Prediction

```python
python3 eval.py --data HN-FB-S --lr 1e-3 --dim 256 --epoch 30 --exp rp --num_layer 2 --num_head 16 --hidden_dim 1024 --dropout 0.2 --smoothing 0.7 --batch_size 2048 --step_size 50 --rp
```

### WikiPeople-

#### Link Prediction

```python
python3 eval.py --data WikiPeople- --lr 1e-3 --dim 256 --epoch 350 --exp lp_nvp --num_layer 3 --num_head 16 --hidden_dim 1024 --dropout 0.2 --smoothing 0.4 --batch_size 2048 --step_size 50 --lp
```

### WD50K

#### Link Prediction

```python
python3 eval.py --data WD50K --lr 1e-3 --dim 256 --epoch 350 --exp lp_nvp --num_layer 3 --num_head 4 --hidden_dim 1024 --dropout 0.2 --smoothing 0.7 --batch_size 2048 --step_size 50 --lp
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
- `--num_layer`: $L_\mathrm{P}=L_\mathrm{C}$
- `--num_head`: $n_\mathrm{P}=n_\mathrm{C}$
- `--hidden_dim`: $d_\mathrm{F}=\hat{d}_\mathrm{F}$
- `--dropout`: $\delta$
- `--smoothing`: $\epsilon$
- `--batch_size`: the batch size
- `--step_size`: the step size of the cosine annealing learning rate scheduler

### Hyperparameters
We tuned HyNT with the following tuning range:
- lr: {0.0005, 0.001}
- dim: 256
- num_epoch: 750
- valid_epoch: 50
- num_layer: {2, 3}
- num_head: {4, 8, 16}
- hidden_dim: 1024
- dropout: {0.1, 0.2}
- smoothing: {0.3, 0.4, 0.5, 0.7}
- batch_size: {1024, 2048} (We fixed the batch_size to 512 for HN-FB, the largest dataset.)
- step_size: {50, 100}

## Generating Datasets

You can download the codes for generating datasets from https://drive.google.com/file/d/1NOThGP_wXdQeGgEGut3Pzn-Jth96jIIs/view?usp=sharing.
