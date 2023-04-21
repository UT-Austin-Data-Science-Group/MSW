import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from skimage.transform import resize
import os
import random
from sklearn import cluster
from tqdm import tqdm
import sys
import torch
import time
import argparse
from utils import *
import ot
np.random.seed(1)
torch.manual_seed(1)

parser = argparse.ArgumentParser(description='CT')
# parser.add_argument('--L', type=int, default=3, metavar='N',
#                     help='input batch size for training (default: 100)')
# parser.add_argument('--k', type=int, default=3, metavar='N',
#                     help='input num batch for training (default: 200)')
parser.add_argument('--num_iter', type=int, default=10000, metavar='N',
                    help='Num Interations')
# parser.add_argument('--T', type=int, default=1, metavar='N',
#                     help='Num Interations')
# parser.add_argument('--N', type=int, default=1, metavar='N',
#                     help='Num Interations')
# parser.add_argument('--M', type=int, default=0, metavar='N',
#                     help='Num Interations')
parser.add_argument('--source', type=str, metavar='N',
                    help='Source')
parser.add_argument('--target', type=str, metavar='N',
                    help='Target')
parser.add_argument('--cluster',  action='store_true',
                    help='Use clustering')
parser.add_argument('--load',  action='store_true',
                    help='Load precomputed')
parser.add_argument('--palette',  action='store_true',
                    help='Show color palette')
# parser.add_argument('--sw_type', type=str, metavar='N',
#                     help='Target')


args = parser.parse_args()


n_clusters = 3000
name1=args.source#path to images 1
name2=args.target#path to images 2
source = img_as_ubyte(io.imread(name1))
target = img_as_ubyte(io.imread(name2))
reshaped_target = img_as_ubyte(resize(target, source.shape[:2]))
name1=name1.replace('/', '')
name2=name2.replace('/', '')
if(args.cluster):
    X = source.reshape((-1, 3))  # We need an (n_sample, n_feature) array
    source_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, n_init=4, batch_size=100)
    source_k_means.fit(X)
    source_values = source_k_means.cluster_centers_.squeeze()
    source_labels = source_k_means.labels_

    # create an array from labels and values
    #source_compressed = np.choose(labels, values)
    source_compressed = source_values[source_labels]
    source_compressed.shape = source.shape

    vmin = source.min()
    vmax = source.max()

    # original image
    plt.figure(1, figsize=(5, 5))
    plt.title("Original Source")
    plt.imshow(source,  vmin=vmin, vmax=256)

    # compressed image
    plt.figure(2, figsize=(5, 5))
    plt.title("Compressed Source")
    plt.imshow(source_compressed.astype('uint8'),  vmin=vmin, vmax=vmax)
    os.makedirs('npzfiles', exist_ok=True)
    with open('npzfiles/'+name1+'source_compressed.npy', 'wb') as f:
        np.save(f, source_compressed)
    with open('npzfiles/'+name1+'source_values.npy', 'wb') as f:
        np.save(f, source_values)
    with open('npzfiles/'+name1+'source_labels.npy', 'wb') as f:
        np.save(f, source_labels)
    np.random.seed(0)

    X = target.reshape((-1, 3))  # We need an (n_sample, n_feature) array
    target_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, n_init=4, batch_size=100)
    target_k_means.fit(X)
    target_values = target_k_means.cluster_centers_.squeeze()
    target_labels = target_k_means.labels_

    # create an array from labels and values
    target_compressed = target_values[target_labels]
    target_compressed.shape = target.shape

    vmin = target.min()
    vmax = target.max()

    # original image
    plt.figure(1, figsize=(5, 5))
    plt.title("Original Target")
    plt.imshow(target,  vmin=vmin, vmax=256)

    # compressed image
    plt.figure(2, figsize=(5, 5))
    plt.title("Compressed Target")
    plt.imshow(target_compressed.astype('uint8'),  vmin=vmin, vmax=vmax)

    with open('npzfiles/'+name2+'target_compressed.npy', 'wb') as f:
        np.save(f, target_compressed)
    with open('npzfiles/'+name2+'target_values.npy', 'wb') as f:
        np.save(f, target_values)
    with open('npzfiles/'+name2+'target_labels.npy', 'wb') as f:
        np.save(f, target_labels)
else:
    with open('npzfiles/'+name1+'source_compressed.npy', 'rb') as f:
        source_compressed = np.load(f)
    with open('npzfiles/'+name2+'target_compressed.npy', 'rb') as f:
        target_compressed = np.load(f)
    with open('npzfiles/'+name1+'source_values.npy', 'rb') as f:
        source_values = np.load(f)
    with open('npzfiles/'+name2+'target_values.npy', 'rb') as f:
        target_values = np.load(f)
    with open('npzfiles/'+name1+'source_labels.npy', 'rb') as f:
        source_labels = np.load(f)

for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)

start = time.time()
SWcluster,SW = transform_SW(source_values,target_values,source_labels,source,L=45,sw_type='sw',num_iter=args.num_iter)
SWtime = np.round(time.time() - start,2)

for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)
start = time.time()
MaxSWcluster,MaxSW = transform_SW(source_values,target_values,source_labels,source,T=45,s_lr=0.1,sw_type='maxsw',num_iter=args.num_iter)
MaxSWtime = np.round(time.time() - start,2)


for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)
start = time.time()
rMSWcluster,rMSW = transform_SW(source_values,target_values,source_labels,source,L=3,T=5,s_lr=0.1,sw_type='rmsw',kappa=50,num_iter=args.num_iter)
rMSWtime = np.round(time.time() - start,2)

for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)
start = time.time()
oMSWcluster,oMSW = transform_SW(source_values,target_values,source_labels,source,L=3,T=5,s_lr=0.1,sw_type='omsw',num_iter=args.num_iter)
oMSWtime = np.round(time.time() - start,2)


for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)
start = time.time()
KSWcluster,KSW = transform_SW(source_values,target_values,source_labels,source,L=15,T=3,sw_type='ksw',num_iter=args.num_iter)
KSWtime = np.round(time.time() - start,2)
for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)
start = time.time()
MaxKSWcluster,MaxKSW = transform_SW(source_values,target_values,source_labels,source,L=3,T=15,s_lr=0.1,sw_type='maxksw',num_iter=args.num_iter)
MaxKSWtime = np.round(time.time() - start,2)
for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)
start = time.time()
iMSWcluster,iMSW = transform_SW(source_values,target_values,source_labels,source,L=3,T=5,s_lr=0.1,sw_type='imsw',num_iter=args.num_iter)
iMSWtime = np.round(time.time() - start,2)
for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)
start = time.time()
viMSWcluster,viMSW = transform_SW(source_values,target_values,source_labels,source,L=3,T=5,s_lr=0.1,kappa=50,sw_type='vimsw',num_iter=args.num_iter)
viMSWtime = np.round(time.time() - start,2)

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
source3=source_values.reshape(-1,3)
reshaped_target3=target_values.reshape(-1,3)

SWcluster=SWcluster/np.max(SWcluster)*255
MaxSWcluster=MaxSWcluster/np.max(MaxSWcluster)*255
KSWcluster=KSWcluster/np.max(KSWcluster)*255
MaxKSWcluster=MaxKSWcluster/np.max(MaxKSWcluster)*255
iMSWcluster=iMSWcluster/np.max(iMSWcluster)*255
rMSWcluster=rMSWcluster/np.max(rMSWcluster)*255
oMSWcluster=oMSWcluster/np.max(oMSWcluster)*255
viMSWcluster=viMSWcluster/np.max(viMSWcluster)*255



# f.suptitle("L={}, k={}, T={}".format(L, k, iter), fontsize=20)
C_SW = ot.dist(SWcluster,reshaped_target3)
C_MaxSW = ot.dist(MaxSWcluster, reshaped_target3)
C_KSW = ot.dist(KSWcluster, reshaped_target3)
C_MaxKSW = ot.dist(MaxKSWcluster, reshaped_target3)
C_iMSW = ot.dist(iMSWcluster, reshaped_target3)
C_rMSW = ot.dist(rMSWcluster, reshaped_target3)
C_oMSW = ot.dist(oMSWcluster, reshaped_target3)
C_viMSW = ot.dist(viMSWcluster, reshaped_target3)

W_SW = np.round(ot.emd2([],[],C_SW),2)
W_MaxSW = np.round(ot.emd2([], [], C_MaxSW),2)
W_KSW = np.round(ot.emd2([], [], C_KSW), 2)
W_MaxKSW = np.round(ot.emd2([], [], C_MaxKSW), 2)
W_iMSW = np.round(ot.emd2([],[],C_iMSW),2)
W_rMSW = np.round(ot.emd2([],[],C_rMSW),2)
W_oMSW = np.round(ot.emd2([],[],C_oMSW),2)
W_viMSW = np.round(ot.emd2([],[],C_viMSW),2)


f, ax = plt.subplots(4, 5, figsize=(12, 5))
ax[0,0].set_title('Source', fontsize=14)
ax[0,0].imshow(source)
ax[1,0].scatter(source3[:, 0], source3[:, 1], source3[:, 2], c=source3 / 255)

ax[0,1].set_title('SW (L={}), {}(s), $W_2={}$'.format(45,SWtime,W_SW), fontsize=8)
ax[0,1].imshow(SW)
ax[1,1].scatter(SWcluster[:, 0], SWcluster[:, 1], SWcluster[:, 2], c=SWcluster / 255)

ax[0,2].set_title('Max-SW (T={}), {}(s), $W_2={}$'.format(45,MaxSWtime,W_MaxSW), fontsize=8)
ax[0,2].imshow(MaxSW)
ax[1,2].scatter(MaxSWcluster[:, 0], MaxSWcluster[:, 1], MaxSWcluster[:, 2], c=MaxSWcluster / 255)

ax[0,3].set_title('K-SW (L={},K={}), {}(s), $W_2={}$'.format(15,3,KSWtime,W_KSW), fontsize=8)
ax[0,3].imshow(KSW)
ax[1,3].scatter(KSWcluster[:, 0], KSWcluster[:, 1], KSWcluster[:, 2], c=KSWcluster / 255)


ax[0,4].set_title('Max-K-SW (K={},T={}), {}(s), $W_2={}$'.format(3,15,MaxKSWtime,W_MaxKSW), fontsize=8)
ax[0,4].imshow(MaxKSW)
ax[1,4].scatter(MaxKSWcluster[:, 0], MaxKSWcluster[:, 1], MaxKSWcluster[:, 2], c=MaxKSWcluster / 255)

ax[2,0].set_title('rMSW (L={},T={},$\kappa$={}), {}(s), $W_2={}$'.format(3,5,50,rMSWtime,W_rMSW), fontsize=8)
ax[2,0].imshow(rMSW)
ax[3,0].scatter(rMSWcluster[:, 0], rMSWcluster[:, 1], rMSWcluster[:, 2], c=rMSWcluster / 255)

ax[2,1].set_title('oMSW (L={},T={}), {}(s), $W_2={}$'.format(3,5,oMSWtime,W_oMSW), fontsize=8)
ax[2,1].imshow(oMSW)
ax[3,1].scatter(oMSWcluster[:, 0], oMSWcluster[:, 1], oMSWcluster[:, 2], c=oMSWcluster / 255)

ax[2,2].set_title('iMSW (L={},T={}), {}(s), $W_2={}$'.format(3,5,iMSWtime,W_iMSW), fontsize=8)
ax[2,2].imshow(iMSW)
ax[3,2].scatter(iMSWcluster[:, 0], iMSWcluster[:, 1], iMSWcluster[:, 2], c=iMSWcluster / 255)

ax[2,3].set_title('viMSW (L={},T={},$\kappa$={}), {}(s), $W_2={}$'.format(3,5,50,viMSWtime,W_viMSW), fontsize=8)
ax[2,3].imshow(viMSW)
ax[3,3].scatter(viMSWcluster[:, 0], viMSWcluster[:, 1], viMSWcluster[:, 2], c=viMSWcluster / 255)


ax[2,4].set_title('Target', fontsize=14)
ax[2,4].imshow(reshaped_target)
ax[3,4].scatter(reshaped_target3[:, 0], reshaped_target3[:, 1], reshaped_target3[:, 2], c=reshaped_target3 / 255)

for i in range(4):
    for j in range(5):
        ax[i,j].get_yaxis().set_visible(False)
        ax[i,j].get_xaxis().set_visible(False)

plt.tight_layout()
plt.subplots_adjust(left=0, right=1, top=0.88, bottom=0.01, wspace=0, hspace=0.145)
plt.show()
plt.clf()
plt.close()
#####
f, ax = plt.subplots(2, 5, figsize=(12, 5))
# f.suptitle("m={}, k={}, T={}".format(m, k, iter), fontsize=20)
ax[0,0].set_title('Source', fontsize=8)
ax[0,0].imshow(source)


ax[0,1].set_title('SW (L={}), {}(s), $W_2={}$'.format(45,SWtime,W_SW), fontsize=8)
ax[0,1].imshow(SW)


ax[0,2].set_title('Max-SW (T={}), {}(s), $W_2={}$'.format(45,MaxSWtime,W_MaxSW), fontsize=8)
ax[0,2].imshow(MaxSW)


ax[0,3].set_title('K-SW (L={},K={}), {}(s), $W_2={}$'.format(15,3,KSWtime,W_KSW), fontsize=8)
ax[0,3].imshow(KSW)



ax[0,4].set_title('Max-K-SW (K={},T={}), {}(s), $W_2={}$'.format(3,15,MaxKSWtime,W_MaxKSW), fontsize=8)
ax[0,4].imshow(MaxKSW)


ax[1,0].set_title('rMSW (L={},T={},$\kappa$={}), {}(s), $W_2={}$'.format(3,5,50,rMSWtime,W_rMSW), fontsize=8)
ax[1,0].imshow(rMSW)
ax[1,1].set_title('oMSW (L={},T={}), {}(s), $W_2={}$'.format(3,5,oMSWtime,W_oMSW), fontsize=8)
ax[1,1].imshow(oMSW)

ax[1,2].set_title('iMSW (L={},T={}), {}(s), $W_2={}$'.format(3,5,iMSWtime,W_iMSW), fontsize=8)
ax[1,2].imshow(iMSW)

ax[1,3].set_title('viMSW (L={},T={},$\kappa$={}), {}(s), $W_2={}$'.format(3,5,50,viMSWtime,W_viMSW), fontsize=8)
ax[1,3].imshow(viMSW)



ax[1,4].set_title('Target', fontsize=8)
ax[1,4].imshow(reshaped_target)


for i in range(2):
    for j in range(5):
        ax[i,j].get_yaxis().set_visible(False)
        ax[i,j].get_xaxis().set_visible(False)

plt.tight_layout()
plt.subplots_adjust(left=0, right=1, top=0.88, bottom=0.01, wspace=0, hspace=0.145)
plt.show()



