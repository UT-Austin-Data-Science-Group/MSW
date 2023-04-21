# Markovian Sliced Wasserstein distance
Python3 implementation of the papers [Markovian Sliced Wasserstein distances: Beyond Indepedent Projections](Markovian sliced wasserstein distances: beyond independent projections)

## Requirement

* python 3.8.8
* pytorch 1.8.1
* torchvision
* numpy
* tqdm
* tensorboardX
* tensorflow-gpu
* imageio

## What is included?
* Gradient flow 
* Color Transfer
* Deep Generative Modeling


## Gradient Flow
```
cd GradientFlow
python main.py
```

## Color Transfer

```
cd ColorTransfer
python main.py --source [source image] --target [target image] --num_iter 2000 --cluster

```

## Deep Generative Modeling
### Code organization
* cfg.py : this file contains arguments for training.
* datasets.py : this file implements dataloaders.
* functions.py : this file implements training functions.
* trainsw.py : this file is the main file for running.
* models : this folder contains neural networks architecture.
* utils : this folder contains implementation of fid score and Inception score.
* fid_stat : this folder contains statistic files for fID score.


### Main path arguments
--dataset : type of dataset {"cifar10","celeba"}
--sw_type : type of distances {"sw","maxsw","ksw","maxksw","rMSW","oMSW","iMSW","viMSW"}
--img_size : size of images
--dis_bs : size of mini-batches
--model : "sngan_{dataset}"
--eval_batch_size : batchsize for computing FID
--L : number of projections 
--K : number of orthogonal projections
--s_lr : slice learning rate (for Max-SW and MSW variants)
--s_max_iter : max iterations of gradient update (for Max-SW and Max-K-SW) and plays as the number of time steps T for MSW variants

### Example

```
 python trainsw.py -gen_bs 128 -dis_bs 128 --data_path ./data --dataset celeba --img_size 64 --max_iter 50000 --model sngan_celeba --latent_dim 128 --gf_dim 256 --df_dim 128 --g_spectral_norm False --d_spectral_norm True --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --n_critic 5 --val_freq 20 --sw_type iMsw --L 10 --s_max_iter 100 --s_lr 0.01 --random_seed 1
```

## Acknowledgment
The structure of this repo is largely based on [sngan.pytorch](https://github.com/GongXinyuu/sngan.pytorch). The implementation of the Von Mises-Fisher distribution is taken from [s-vae-pytorch](https://github.com/nicola-decao/s-vae-pytorch).