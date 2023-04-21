"""
Gradient flows in 2D
====================

Let's showcase the properties of **kernel MMDs**, **Hausdorff**
and **Sinkhorn** divergences on a simple toy problem:
the registration of one blob onto another.
"""
import ot

##############################################
# Setup
# ---------------------
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import cost_matrix
import torch
import numpy as np
import torch
from random import choices
from imageio import imread
from matplotlib import pyplot as plt
from von_mises_fisher import VonMisesFisher
# import cvxpy as cp
# from geomloss import SamplesLoss
np.random.seed(1)
torch.manual_seed(1)
random.seed(1)

def compute_true_Wasserstein(X,Y,p=2):
    M = ot.dist(X.detach().numpy(), Y.detach().numpy())
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)
def compute_Wasserstein(M,device='cpu',e=0):
    if(e==0):
        pi = ot.emd([],[],M.cpu().detach().numpy()).astype('float32')
    else:
        pi = ot.sinkhorn([], [], M.cpu().detach().numpy(),reg=e).astype('float32')
    pi = torch.from_numpy(pi).to(device)
    return torch.sum(pi*M)

def rand_projections(dim, num_projections=1000,device='cpu'):
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
    wasserstein_distance = torch.pow(wasserstein_distance, p).mean()
    return wasserstein_distance

def SW(X, Y, L=30, p=2, device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim, L,device)
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    sw=one_dimensional_Wasserstein_prod(X_prod,Y_prod,p=p)
    return  torch.pow(sw,1./p)




def MaxSW(X,Y,p=2,s_lr=0.1,n_lr=30,device="cpu"):
    dim = X.size(1)
    theta = torch.randn((1, dim), device=device, requires_grad=True)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1,keepdim=True))
    optimizer = torch.optim.SGD([theta], lr=s_lr)
    X_detach = X.detach()
    Y_detach = Y.detach()
    for _ in range(n_lr-1):
        X_prod = torch.matmul(X_detach, theta.transpose(0, 1))
        Y_prod = torch.matmul(Y_detach, theta.transpose(0, 1))
        negative_sw = -torch.pow(one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p),1./p)
        optimizer.zero_grad()
        negative_sw.backward()
        optimizer.step()
        theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1,keepdim=True))
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    sw = one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p)
    return torch.pow(sw,1./p)

def KSW(X, Y, L=15,n_lr=2, p=2, device="cpu"):
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
    theta = torch.cat(thetas, dim=0)
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    sw=one_dimensional_Wasserstein_prod(X_prod,Y_prod,p=p)
    return  torch.pow(sw,1./p)


def oMSW(X, Y, L=5,n_lr=2, p=2, device="cpu"):
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
    return torch.pow(sw, 1. / p)


def rMSW(X, Y, L=2,kappa=50,n_lr=5, p=2, device="cpu"):
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
    return  torch.pow(sw,1./p)
def projection(U, V):
    return torch.sum(V * U,dim=1,keepdim=True)* U / torch.sum(U * U,dim=1,keepdim=True)
def MaxKSW(X,Y,L=2,p=2,s_lr=0.1,n_lr=15,device="cpu"):
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
        negative_sw = -torch.pow(one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p),1./p)
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
    return torch.pow(sw,1./p)









use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

###############################################
# Display routine
# ~~~~~~~~~~~~~~~~~





def load_image(fname):
    img = imread(fname, as_gray=True)  # Grayscale
    img = (img[::-1, :]) / 255.
    return 1 - img


def draw_samples(fname, n, dtype=torch.FloatTensor):
    A = load_image(fname)
    xg, yg = np.meshgrid(np.linspace(0, 1, A.shape[0]), np.linspace(0, 1, A.shape[1]))

    grid = list(zip(xg.ravel(), yg.ravel()))
    dens = A.ravel() / A.sum()
    dots = np.array(choices(grid, dens, k=n))
    dots += (.5 / A.shape[0]) * np.random.standard_normal(dots.shape)

    return torch.from_numpy(dots).type(dtype)


def display_samples(ax, x, color):
    x_ = x.detach().cpu().numpy()
    ax.scatter(x_[:, 0], x_[:, 1], 25 * 500 / len(x_), color, edgecolors='none')



np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
N, M = (1000, 1000) if not use_cuda else (1000, 1000)

X_i = draw_samples("density_a.png", N, dtype)
Y_j = draw_samples("density_b.png", M, dtype)

def iMSW(X,Y,L=2,p=2,s_lr=0.1,n_lr=5,M=0,N=1,device="cpu",ortho_type="normal"):
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
        negative_sw = -torch.pow(one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p),1./p)
        optimizer.zero_grad()
        negative_sw.backward()
        optimizer.step()
        theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1, keepdim=True))
        thetas.append(theta.data)
    theta = torch.cat(thetas[M:][::N],dim=0)
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    sw = one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p)
    return torch.pow(sw,1./p)
def viMSW(X,Y,L=2,kappa=50,p=2,s_lr=0.1,n_lr=5,M=0,N=1,device="cpu",ortho_type="normal"):
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
        negative_sw = -torch.pow(one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p),1./p)
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
    return torch.pow(sw,1./p)
def gradient_flow(loss, lr=.0001,title='m-OT',flag=False) :
    """Flows along the gradient of the cost function, using a simple Euler scheme.

    Parameters:
        loss ((x_i,y_j) -> torch float number):
            Real-valued loss function.
        lr (float, default = .05):
            Learning rate, i.e. time step.
    """
    for _ in range(1000):
        a = np.random.randn(100)
        a = torch.randn(100)
    # Parameters for the gradient descent
    Nsteps = int(0.05/lr)+1
    display_its = [int(t/lr) for t in [0,0.02,0.03]] #[0, 0.01,0.015,0.02,0.025,0.03]]

    # Use colors to identify the particles
    colors = (10*X_i[:,0]).cos() * (10*X_i[:,1]).cos()
    colors = colors.detach().cpu().numpy()

    # Make sure that we won't modify the reference samples
    x_i, y_j = X_i.clone(), Y_j.clone()

    # We're going to perform gradient descent on Loss(α, β)
    # wrt. the positions x_i of the diracs masses that make up α:
    x_i.requires_grad = True

    t_0 = time.time()
    # plt.figure(figsize=(12,8)) ; k = 1
    plt.figure(figsize=(8, 6));
    k = 1
    start = time.time()
    for i in range(Nsteps): # Euler scheme ===============
        # Compute cost and gradient
        L_αβ = loss(x_i, y_j)
        [g]  = torch.autograd.grad(L_αβ, [x_i])

        if i in display_its : # display
            # ax = plt.subplot(1,6,k) ; k = k+1
            ax = plt.subplot(1,3,k) ; k = k+1

            if(i==0):
                ax.set_ylabel(title,fontsize=11)
            plt.set_cmap("hsv")
            plt.scatter( [10], [10] ) # shameless hack to prevent a slight change of axis...

            display_samples(ax, y_j, [(.55,.55,.95)])
            display_samples(ax, x_i, colors)
            if(i!=0):
                time_ms = np.round(time.time()-start,2)
            else:
                time_ms =0
            ax.set_title("$W_2$: "+str(np.round(compute_true_Wasserstein(x_i.cpu(),y_j.cpu())*100,4)) +r"$\times 10^{-2}$"+" ("+str(time_ms)+"s)",fontsize=11)
            if(flag):
                ax.set_xlabel("steps "+str(i),fontsize=11)
            plt.axis([0,1,0,1])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xticks([], []); plt.yticks([], [])
            plt.tight_layout()

        # in-place modification of the tensor's values
        x_i.data -= lr * len(x_i) * g
    # plt.title("t = {:1.2f}, elapsed time: {:.2f}s/it".format(lr*i, (time.time() - t_0)/Nsteps ))
    plt.subplots_adjust(left=0.03, bottom=0, right=0.99, top=0.91, wspace=0, hspace=0.2)
    plt.show()

#warm up
for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)
np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
gradient_flow(SW,title='SW L=30',flag=True)
#
np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
gradient_flow(MaxSW,title='Max-SW T=30',flag=False)
#
np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
gradient_flow(iMSW,title='iMSW L=2 T=5',flag=False)
# 
np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
gradient_flow(KSW,title='K-SW L=15 K=2',flag=False)
# 
np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
gradient_flow(MaxKSW,title='Max-K-SW K=2 T=15',flag=False)


# np.random.seed(1)
# torch.manual_seed(1)
# random.seed(1)
# gradient_flow(oMSW,title='oMSW L=5 T=2',flag=False)

np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
gradient_flow(viMSW,title='viMSW L=2 T=5 $\kappa$=50',flag=False)

