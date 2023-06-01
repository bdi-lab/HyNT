# Representation Learning on Hyper-Relational and Numeric Knowledge Graphs with Transformers
This code is the official implementation of the following [paper](https://arxiv.org/abs/2305.18256):

> Chanyoung Chung, Jaejun Lee, and Joyce Jiyoung Whang, Representation Learning on Hyper-Relational and Numeric Knowledge Graphs with Transformers, To appear in the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2023.

Codes are written by Jaejun Lee (jjlee98@kaist.ac.kr). When you use this code or data, please cite our paper.
```bibtex
@article{hynt,
	author={Chanyoung Chung and Jaejun Lee and Joyce Jiyoung Whang},
	title={Representation Learning on Hyper-Relational and Numeric Knowledge Graphs with Transformers},
	year={2023},
	journal={arXiv preprint arXiv:2305.18256},
	doi={10.48550/arXiv.2305.18256}
}
```

## Requirements

We used python 3.7 and PyTorch 1.12.0 with cudatoolkit 11.3.

You can install all requirements with:

```shell
pip install -r requirements.txt
```

## Reproducing Results

We used NVIDIA RTX A6000 and NVIDIA GeForce RTX 3090 for all our experiments.

We provide the checkpoints we used to produce the link prediction results, relation prediction results, and numeric value prediction results on HN-WK, HN-YG, HN-FB, and HN-FB-S. The checkpoints we used to produce the link prediction results on WD50K and WikiPeople<sup>$\mathbf{-}$</sup> are also provided.

You can download the checkpoints in https://drive.google.com/file/d/1EUg7n5vsfnrT-R0B6851y7RJvjeWYTyo/view?usp=sharing.

For usage, place the unzipped checkpoint folder in the same directory with the codes.

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

To train HyNT from scratch, run `train.py` with arguments.

The list of arguments of 'train.py' and their brief descriptions:
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

Please refer to `train.py` or `eval.py` for the examples of the arguments.

### Hyperparameters
We tuned HyNT with the following (except HN-FB, where we fixed the batch_size to 512):
- lr: {0.0005, 0.001}
- dim: 256
- num_epoch: 750
- valid_epoch: 50
- num_layer: {2, 3}
- num_head: {4, 8, 16}
- hidden_dim: 1024
- dropout: {0.1, 0.2}
- smoothing: {0.3, 0.5, 0.7}
- batch_size: {1024, 2048}
- step_size: {50, 100}
