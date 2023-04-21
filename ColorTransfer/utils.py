import numpy as np
import ot
import random
import tqdm
import torch
from von_mises_fisher import VonMisesFisher
def compute_true_Wasserstein(X,Y,p=2):
    M = ot.dist(X.detach().numpy(), Y.detach().numpy())
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)

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




# def MaxSW(X,Y,p=2,s_lr=0.1,n_lr=10,device="cpu"):
#     dim = X.size(1)
#     theta = torch.randn((1, dim), device=device, requires_grad=True)
#     theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1,keepdim=True))
#     optimizer = torch.optim.SGD([theta], lr=s_lr)
#     X_detach = X.detach()
#     Y_detach = Y.detach()
#     for _ in range(n_lr-1):
#         X_prod = torch.matmul(X_detach, theta.transpose(0, 1))
#         Y_prod = torch.matmul(Y_detach, theta.transpose(0, 1))
#         negative_sw = -torch.pow(one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p).mean(),1./p)
#         optimizer.zero_grad()
#         negative_sw.backward()
#         optimizer.step()
#         theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1,keepdim=True))
#     X_prod = torch.matmul(X, theta.transpose(0, 1))
#     Y_prod = torch.matmul(Y, theta.transpose(0, 1))
#     sw = one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p)
#     return torch.pow(sw.mean(),1./p)
#
#
#
#
# def MaxHSW(X,Y,k,p=2,s_lr=0.01,n_lr=10,device="cpu"):
#     dim = X.size(1)
#     theta1 = torch.randn((k, dim), device=device, requires_grad=True)
#     theta1.data = theta1.data / torch.sqrt(torch.sum(theta1.data ** 2, dim=1,keepdim=True))
#     theta2 = torch.randn((1, k), device=device, requires_grad=True)
#     theta2.data = theta2.data / torch.sqrt(torch.sum(theta2.data ** 2, dim=1,keepdim=True))
#     optimizer = torch.optim.SGD([theta1,theta2], lr=s_lr)
#     X_detach = X.detach()
#     Y_detach = Y.detach()
#     for _ in range(n_lr-1):
#         X_prod = torch.matmul(torch.matmul(X_detach, theta1.transpose(0, 1)), theta2.transpose(0, 1))
#         Y_prod = torch.matmul(torch.matmul(Y_detach, theta1.transpose(0, 1)), theta2.transpose(0, 1))
#         negative_hsw = -torch.pow(one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p).mean(),1./p)
#         optimizer.zero_grad()
#         negative_hsw.backward()
#         optimizer.step()
#         theta1.data = theta1.data / torch.sqrt(torch.sum(theta1.data ** 2, dim=1,keepdim=True))
#         theta2.data = theta2.data / torch.sqrt(torch.sum(theta2.data ** 2, dim=1,keepdim=True))
#     X_prod = torch.matmul(torch.matmul(X, theta1.transpose(0, 1)), theta2.transpose(0, 1))
#     Y_prod = torch.matmul(torch.matmul(Y, theta1.transpose(0, 1)), theta2.transpose(0, 1))
#     wasserstein_distance = one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p)
#     return torch.pow(wasserstein_distance.mean(), 1. / p)
def get_powers(dim, degree):
    '''
    This function calculates the powers of a homogeneous polynomial
    e.g.
    list(get_powers(dim=2,degree=3))
    [(0, 3), (1, 2), (2, 1), (3, 0)]
    list(get_powers(dim=3,degree=2))
    [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
    '''
    if dim == 1:
        yield (degree,)
    else:
        for value in range(degree + 1):
            for permutation in get_powers(dim - 1, degree - value):
                yield (value,) + permutation

def homopoly(dim, degree):
    '''
    calculates the number of elements in a homogeneous polynomial
    '''
    return len(list(get_powers(dim, degree)))

def poly( X, theta,degree):
    ''' The polynomial defining function for generalized Radon transform
        Inputs
        X:  Nxd matrix of N data samples
        theta: Lxd vector that parameterizes for L projections
        degree: degree of the polynomial
    '''
    N, d = X.shape
    assert theta.shape[1] == homopoly(d, degree)
    powers = list(get_powers(d, degree))
    HX = torch.ones((N, len(powers)))
    for k, power in enumerate(powers):
        for i, p in enumerate(power):
            HX[:, k] *= X[:, i] ** p
    return torch.matmul(HX, theta.T)

def GSW(X, Y, L=10, p=2, device="cpu"):
    dim = X.size(1)
    dpoly = homopoly(dim, 3)
    theta = rand_projections(dpoly, L,device)
    X_prod = poly(X,theta,3)
    Y_prod = poly(Y,theta,3)
    sw=one_dimensional_Wasserstein_prod(X_prod,Y_prod,p=p)
    return  torch.pow(sw.mean(),1./p)
def SW(X, Y, L=10, p=2, device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim, L,device)
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    sw=one_dimensional_Wasserstein_prod(X_prod,Y_prod,p=p)
    return  torch.pow(sw,1./p)


def MaxSW(X,Y,p=2,s_lr=0.1,n_lr=10,device="cpu"):
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

def KSW(X, Y, L=5,n_lr=2, p=2, device="cpu"):
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


def MarkovorthoSW(X, Y, L=5,n_lr=2, p=2, device="cpu"):
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

def MarkovorthoVMFSW(X, Y, L=5,kappa=50,n_lr=2, p=2, device="cpu"):
    dim = X.size(1)
    theta = torch.randn((L, dim), device=device)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1,keepdim=True))
    thetas = [theta.data]
    for _ in range(n_lr - 1):
        new_theta = torch.randn((L, dim), device=device)
        new_theta = new_theta - projection(thetas[-1], new_theta)
        new_theta = new_theta / torch.sqrt(torch.sum(new_theta ** 2, dim=1,keepdim=True))
        vmf = VonMisesFisher(new_theta, torch.full((L, 1), kappa, device=device))
        theta.data = vmf.rsample(1).view(L, -1)
        thetas.append(new_theta)
    theta = torch.cat(thetas, dim=0)
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    sw = one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=p)
    return torch.pow(sw, 1. / p)

def MarkovVMFSW(X, Y, L=2,kappa=10,n_lr=5, p=2, device="cpu"):
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
def MaxKSW(X,Y,L=2,p=2,s_lr=0.1,n_lr=5,device="cpu"):
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
def MarkovMaxSW(X,Y,L=2,p=2,s_lr=0.1,n_lr=5,M=0,N=1,device="cpu",ortho_type="normal"):
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
def MarkovMaxVMFSW(X,Y,L=2,kappa=50,p=2,s_lr=0.1,n_lr=5,M=0,N=1,device="cpu",ortho_type="normal"):
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
def HSW(X, Y,L=10,k=6,p=2, device="cpu"):
    dim = X.size(1)
    theta1 = rand_projections(dim, k,device)
    theta2 = rand_projections(k, L,device)
    X_prod = torch.matmul(torch.matmul(X, theta1.transpose(0, 1)),theta2.transpose(0, 1))
    Y_prod = torch.matmul(torch.matmul(Y, theta1.transpose(0, 1)),theta2.transpose(0, 1))
    hsw=one_dimensional_Wasserstein_prod(X_prod,Y_prod,p=p)
    return  torch.pow(hsw.mean(),1./p)
def HGSW(X, Y,L=10,k=6,p=2, device="cpu"):
    dim = X.size(1)
    dpoly = homopoly(dim, 3)
    theta1 = rand_projections(dpoly, k,device)
    X_prod = poly(X, theta1, 3)
    Y_prod = poly(Y, theta1, 3)
    theta2 = rand_projections(k, L,device)
    X_prod = torch.matmul(X_prod, theta2.transpose(0, 1))
    Y_prod = torch.matmul(Y_prod, theta2.transpose(0, 1))
    hsw=one_dimensional_Wasserstein_prod(X_prod,Y_prod,p=p)
    return  torch.pow(hsw.mean(),1./p)

def MarkovNPSW(X, Y, L=10, p=2,rho=3, device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim, L,device)
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_prod, dim=0)[0]
                - torch.sort(Y_prod, dim=0)[0]
        )
    )
    wasserstein_distances = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=0), 1.0 / p)
    wasserstein_distances = torch.pow(wasserstein_distances, p)
    weights = torch.softmax(wasserstein_distances**rho,dim=-1)
    sw = torch.sum(weights*wasserstein_distances)
    return  torch.pow(sw,1./p)

def transform_SW(src,target,src_label,origin,sw_type='SW',L=10,k=2,T=10,s_lr=0.1,kappa=50,M=0,N=1,num_iter=1000):
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    s = np.array(src).reshape(-1, 3)
    s = torch.from_numpy(s).float()
    s = torch.nn.parameter.Parameter(s)
    t = np.array(target).reshape(-1, 3)
    t = torch.from_numpy(t).float()
    opt = torch.optim.SGD([s], lr=0.001)
    for _ in range(num_iter):
        opt.zero_grad()
        if (sw_type == 'sw'):
            g_loss = SW(s, t, L=L)
        elif (sw_type == 'hsw'):
            g_loss = HSW(s, t, k=k, L=L)
        # elif (args.sw_type == 'msw'):
        #     g_loss = MSW(s, t, k=args.k, L=L)
        elif (sw_type == 'imsw'):
            g_loss = MarkovMaxSW(s, t, L=L, s_lr=s_lr, n_lr=T,
                                 M=M, N=N)
        elif (sw_type == 'vimsw'):
            g_loss = MarkovMaxVMFSW(s, t, L=L, kappa=kappa, s_lr=s_lr, n_lr=T,
                                     M=M, N=N)
        elif (sw_type == 'maxsw'):
            g_loss = MaxSW(s, t, s_lr=s_lr, n_lr=T)
        elif (sw_type == 'maxksw'):
            g_loss = MaxKSW(s, t, L=L, s_lr=s_lr, n_lr=T)
        elif (sw_type == 'ksw'):
            g_loss = KSW(s, t, L=L, n_lr=T)
        elif (sw_type == 'rmsw'):
            g_loss = MarkovVMFSW(s, t, L=L, kappa=kappa, n_lr=T)
        elif (sw_type == 'omsw'):
            g_loss = MarkovorthoSW(s, t, L=L, n_lr=T)
        elif (sw_type == 'markovorthovmfsw'):
            g_loss = MarkovorthoVMFSW(s, t, L=L, kappa=kappa, n_lr=T)
        elif (sw_type=='npsw'):
            g_loss= MarkovNPSW(s,t,L=L,rho=kappa)
        g_loss = g_loss*s.shape[0]
        opt.zero_grad()
        g_loss.backward()
        opt.step()
        s.data = torch.clamp(s, min=0)
    s = torch.clamp( s,min=0).cpu().detach().numpy()
    img_ot_transf = s[src_label].reshape(origin.shape)
    img_ot_transf = img_ot_transf / np.max(img_ot_transf) * 255
    img_ot_transf = img_ot_transf.astype("uint8")
    return s, img_ot_transf

# def transform_GSW(src,target,src_label,origin,L, num_iter):
#     np.random.seed(1)
#     random.seed(1)
#     torch.manual_seed(1)
#     s = np.array(src).reshape(-1, 3)/255
#     s = torch.from_numpy(s).float()
#     s = torch.nn.parameter.Parameter(s)
#     t = np.array(target).reshape(-1, 3)/255
#     t = torch.from_numpy(t).float()
#     opt = torch.optim.SGD([s], lr=0.1)
#     for _ in range(num_iter):
#         opt.zero_grad()
#         loss = GSW(s,t,L=L)
#         loss.backward()
#         opt.step()
#     s = torch.clamp( s,min=0).cpu().detach().numpy()
#     img_ot_transf = s[src_label].reshape(origin.shape)
#     img_ot_transf = img_ot_transf / np.max(img_ot_transf) * 255
#     img_ot_transf = img_ot_transf.astype("uint8")
#     return s, img_ot_transf
#
# def transform_HSW(src,target,src_label,origin,L,k, num_iter):
#     np.random.seed(1)
#     random.seed(1)
#     torch.manual_seed(1)
#     s = np.array(src).reshape(-1, 3)/255
#     s = torch.from_numpy(s).float()
#     s = torch.nn.parameter.Parameter(s)
#     t = np.array(target).reshape(-1, 3)/255
#     t = torch.from_numpy(t).float()
#     opt = torch.optim.SGD([s], lr=0.1)
#     for _ in range(num_iter):
#         opt.zero_grad()
#         loss = HSW(s, t, L=L,k=k)
#         loss.backward()
#         opt.step()
#     s =torch.clamp( s,min=0).cpu().detach().numpy()
#     img_ot_transf = s[src_label].reshape(origin.shape)
#     img_ot_transf = img_ot_transf / np.max(img_ot_transf) * 255
#     img_ot_transf = img_ot_transf.astype("uint8")
#     return s, img_ot_transf
#
# def transform_HGSW(src,target,src_label,origin,L,k, num_iter):
#     np.random.seed(1)
#     random.seed(1)
#     torch.manual_seed(1)
#     s = np.array(src).reshape(-1, 3)/255
#     s = torch.from_numpy(s).float()
#     s = torch.nn.parameter.Parameter(s)
#     t = np.array(target).reshape(-1, 3)/255
#     t = torch.from_numpy(t).float()
#     opt = torch.optim.SGD([s], lr=0.1)
#     for _ in range(num_iter):
#         opt.zero_grad()
#         loss = HGSW(s, t, L=L,k=k)
#         loss.backward()
#         opt.step()
#     s =torch.clamp( s,min=0).cpu().detach().numpy()
#     img_ot_transf = s[src_label].reshape(origin.shape)
#     img_ot_transf = img_ot_transf / np.max(img_ot_transf) * 255
#     img_ot_transf = img_ot_transf.astype("uint8")
#     return s, img_ot_transf