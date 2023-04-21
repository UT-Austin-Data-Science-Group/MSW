# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from imageio import imsave
from tqdm import tqdm
from copy import deepcopy
import logging
from utils.inception_score import get_inception_score
from utils.fid_score import calculate_fid_given_paths
from von_mises_fisher import VonMisesFisher
logger = logging.getLogger(__name__)


def train(args, gen_net: nn.Module, dis_net: nn.Module,  gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, schedulers=None,cal_time=False):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        true_bs = imgs.shape[0]
        if (true_bs < args.gen_batch_size):
            break
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)
        true_bs = imgs.shape[0]
        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (true_bs, args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        _,real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        _,fake_validity = dis_net(fake_imgs)
        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()
        dis_optimizer.step()
        writer.add_scalar('d_loss', d_loss.item(), global_steps)



        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            real_feature,realout= dis_net(real_imgs)
            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (true_bs, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_feature,fakeoutput = dis_net(gen_imgs)
            X = torch.cat([real_feature.view(real_feature.shape[0],-1),realout.view(real_feature.shape[0],-1) ],dim=1)
            Y = torch.cat([fake_feature.view(fake_feature.shape[0], -1), fakeoutput.view(fake_feature.shape[0], -1)],
                          dim=1)
            if (cal_time):
                start = time.time()
            if (args.sw_type == 'sw'):
                g_loss =  SW(X, Y, L=args.L)
            elif (args.sw_type == 'iMSW'):
                g_loss =  iMSW(X, Y,L=args.L, s_lr=args.s_lr, n_lr=args.s_max_iter,ortho_type=args.ortho_type,M=args.M,N=args.N)
            elif (args.sw_type == 'viMSW'):
                g_loss =  viMSW(X, Y,L=args.L,kappa=args.kappa, s_lr=args.s_lr, n_lr=args.s_max_iter,ortho_type=args.ortho_type,M=args.M,N=args.N)
            elif (args.sw_type == 'maxsw'):
                g_loss =  MaxSW(X, Y, s_lr=args.s_lr, n_lr=args.s_max_iter)
            elif (args.sw_type == 'maxksw'):
                g_loss =  MaxKSW(X, Y, L=args.L,s_lr=args.s_lr, n_lr=args.s_max_iter)
            elif (args.sw_type == 'ksw'):
                g_loss =  KSW(X, Y, L=args.L,n_lr = args.s_max_iter)
            elif (args.sw_type == 'rMSW'):
                g_loss =  rMSW(X, Y, L=args.L,kappa=args.kappa,n_lr = args.s_max_iter)
            elif (args.sw_type == 'oMSW'):
                g_loss =  oMSW(X, Y, L=args.L,n_lr = args.s_max_iter)
            gen_optimizer.zero_grad()
            g_loss.backward()
            gen_optimizer.step()
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1
            # cal loss

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight


        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1


def validate(args, fixed_z, fid_stat, gen_net: nn.Module, writer_dict):
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']

    # eval mode
    gen_net = gen_net.eval()

    # generate images
    sample_imgs = gen_net(fixed_z)
    img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)

    # get fid and inception score
    fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
    os.makedirs(fid_buffer_dir)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))

    # get inception score
    logger.info('=> calculate inception score')
    if args.dataset.lower() == 'lsun_church' or args.dataset.lower() == 'celebahq' :
        mean, std = get_inception_score(img_list,bs=args.eval_batch_size)
    elif args.dataset.lower() == 'stl10':
        mean, std = get_inception_score(img_list,bs=args.eval_batch_size)
    else:
        mean, std = get_inception_score(img_list,bs=args.eval_batch_size)

    # get fid score
    logger.info('=> calculate fid score')
    if args.dataset.lower() == 'lsun_church' or args.dataset.lower() == 'celebahq' :
        fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None,batch_size=args.eval_batch_size)
    elif args.dataset.lower() == 'stl10':
        fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None,batch_size=args.eval_batch_size)
    else:
        fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None,batch_size=args.eval_batch_size)
    os.system('rm -r {}'.format(fid_buffer_dir))

    writer.add_image('sampled_images', img_grid, global_steps)
    writer.add_scalar('Inception_score/mean', mean, global_steps)
    writer.add_scalar('Inception_score/std', std, global_steps)
    writer.add_scalar('FID_score', fid_score, global_steps)

    writer_dict['valid_global_steps'] = global_steps + 1

    return mean, fid_score


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def rand_projections(dim, num_projections=1000,device='cuda'):
    projections = torch.randn((num_projections, dim),device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections


def one_dimensional_Wasserstein_prod(X_prod,Y_prod,p):
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_prod, dim=0)[0]
                - torch.sort(Y_prod, dim=0)[0]
        )
    )
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=0), 1.0 / p)
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    return wasserstein_distance


def SW(X, Y, L=1000, p=2, device="cuda"):
    dim = X.size(1)
    theta = rand_projections(dim, L,device)
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    sw=one_dimensional_Wasserstein_prod(X_prod,Y_prod,p=p)
    return  torch.pow(sw.mean(),1./p)


def MaxSW(X,Y,p=2,s_lr=0.01,n_lr=10,device="cuda"):
    dim = X.size(1)
    theta = torch.randn((1, dim), device=device, requires_grad=True)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1,keepdim=True))
    optimizer = torch.optim.SGD([theta], lr=s_lr)
    X_detach = X.detach()
    Y_detach = Y.detach()
    for _ in range(n_lr-1):
        X_prod = torch.matmul(X_detach, theta.transpose(0, 1))
        Y_prod = torch.matmul(Y_detach, theta.transpose(0, 1))
        negative_sw = -torch.pow(one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p).mean(),1./p)
        optimizer.zero_grad()
        negative_sw.backward()
        optimizer.step()
        theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1,keepdim=True))
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    sw = one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p)
    return torch.pow(sw.mean(),1./p)

def KSW(X, Y, L=1000,n_lr=10, p=2, device="cuda"):
    dim = X.size(1)
    theta = torch.randn((L, dim),device=device)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1,keepdim=True))
    thetas = [theta.data]
    for _ in range(n_lr-1):
        new_theta = torch.randn((L, dim), device=device)
        for prev_theta in thetas:
            new_theta = new_theta - projection(prev_theta,new_theta)
        new_theta = new_theta / torch.sqrt(torch.sum(new_theta ** 2, dim=1,keepdim=True))
        theta.data=new_theta
        thetas.append(new_theta)
    theta = torch.cat(thetas,dim=0)
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    sw=one_dimensional_Wasserstein_prod(X_prod,Y_prod,p=p)
    return  torch.pow(sw.mean(),1./p)


def oMSW(X, Y, L=1000,n_lr=10, p=2, device="cuda"):
    dim = X.size(1)
    theta = torch.randn((L, dim), device=device)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1,keepdim=True))
    thetas = [theta.data]
    for _ in range(n_lr - 1):
        new_theta = torch.randn((L, dim), device=device)
        new_theta = new_theta - projection(thetas[-1], new_theta)
        new_theta = new_theta / torch.sqrt(torch.sum(new_theta ** 2, dim=1,keepdim=True))
        theta.data = new_theta
        thetas.append(new_theta)
    theta = torch.cat(thetas, dim=0)
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    sw = one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p)
    return torch.pow(sw.mean(), 1. / p)


def rMSW(X, Y, L=1000,kappa=50,n_lr=10, p=2, device="cuda"):
    dim = X.size(1)
    theta = torch.randn((L, dim),device=device)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1,keepdim=True))
    thetas=[theta.data]
    for _ in range(n_lr-1):
        vmf = VonMisesFisher(theta, torch.full((L, 1), kappa, device=device))
        theta.data = vmf.rsample(1).view(L, -1)
        thetas.append(theta.data)
    theta = torch.cat(thetas,dim=0)
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    sw=one_dimensional_Wasserstein_prod(X_prod,Y_prod,p=p)
    return  torch.pow(sw.mean(),1./p)
def projection(U, V):
    return torch.sum(V * U,dim=1,keepdim=True)* U / torch.sum(U * U,dim=1,keepdim=True)
def MaxKSW(X,Y,L,p=2,s_lr=0.01,n_lr=10,device="cuda"):
    dim = X.size(1)
    theta = torch.randn((L, dim), device=device, requires_grad=True)
    theta.data[[0]] = theta.data[[0]] / torch.sqrt(torch.sum(theta.data[[0]] ** 2, dim=1,keepdim=True))
    for l in range(1,L):
        for i in range(l):
            theta.data[[l]]=theta.data[[l]]-projection(theta.data[[i]],theta.data[[l]])
        theta.data[[l]] = theta.data[[l]] / torch.sqrt(torch.sum(theta.data[[l]] ** 2, dim=1,keepdim=True))
    optimizer = torch.optim.SGD([theta], lr=s_lr)
    X_detach = X.detach()
    Y_detach = Y.detach()
    for _ in range(n_lr-1):
        X_prod = torch.matmul(X_detach, theta.transpose(0, 1))
        Y_prod = torch.matmul(Y_detach, theta.transpose(0, 1))
        negative_sw = -torch.pow(one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p).mean(),1./p)
        optimizer.zero_grad()
        negative_sw.backward()
        optimizer.step()
        theta.data[[0]] = theta.data[[0]] / torch.sqrt(torch.sum(theta.data[[0]] ** 2, dim=1,keepdim=True))
        for l in range(1, L):
            for i in range(l):
                theta.data[[l]] = theta.data[[l]] - projection(theta.data[[i]], theta.data[[l]])
            theta.data[[l]] = theta.data[[l]] / torch.sqrt(torch.sum(theta.data[[l]] ** 2, dim=1,keepdim=True))
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    sw = one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p)
    return torch.pow(sw.mean(),1./p)





def iMSW(X,Y,L,p=2,s_lr=0.01,n_lr=10,M=0,N=1,device="cuda",ortho_type="normal"):
    dim = X.size(1)
    theta = torch.randn((L, dim), device=device, requires_grad=True)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1, keepdim=True))
    thetas =[theta.data]
    optimizer = torch.optim.SGD([theta], lr=s_lr)
    X_detach = X.detach()
    Y_detach = Y.detach()
    for _ in range(n_lr-1):
        X_prod = torch.matmul(X_detach, theta.transpose(0, 1))
        Y_prod = torch.matmul(Y_detach, theta.transpose(0, 1))
        negative_sw = -torch.pow(one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p).mean(),1./p)
        optimizer.zero_grad()
        negative_sw.backward()
        optimizer.step()
        theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1, keepdim=True))
        thetas.append(theta.data)
    theta = torch.cat(thetas[M:][::N],dim=0)
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    sw = one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p)
    return torch.pow(sw.mean(),1./p)

def viMSW(X,Y,L,kappa,p=2,s_lr=0.01,n_lr=10,M=0,N=1,device="cuda",ortho_type="normal"):
    dim = X.size(1)
    theta = torch.randn((L, dim), device=device, requires_grad=True)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1, keepdim=True))
    thetas =[theta.data]
    optimizer = torch.optim.SGD([theta], lr=s_lr)
    X_detach = X.detach()
    Y_detach = Y.detach()
    for _ in range(n_lr-1):
        X_prod = torch.matmul(X_detach, theta.transpose(0, 1))
        Y_prod = torch.matmul(Y_detach, theta.transpose(0, 1))
        negative_sw = -torch.pow(one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p).mean(),1./p)
        optimizer.zero_grad()
        negative_sw.backward()
        optimizer.step()
        theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1, keepdim=True))
        vmf = VonMisesFisher(theta, torch.full((L, 1), kappa, device=device))
        theta.data =vmf.rsample(1).view(L,-1)
        thetas.append(theta.data)
    theta = torch.cat(thetas[M:][::N],dim=0)
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    sw = one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p)
    return torch.pow(sw.mean(),1./p)





