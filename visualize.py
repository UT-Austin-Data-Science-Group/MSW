from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from von_mises_fisher import VonMisesFisher
import matplotlib.pyplot as plt
import numpy as np
import torch
# Fixing random state for reproducibility
np.random.seed(19680801)


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100
a = np.random.randn(n,3)
a = a/np.sqrt(np.sum(a**2,axis=1,keepdims=True))
mu = torch.Tensor([[0,1,-1]])
mu = mu/torch.sqrt(torch.sum(mu**2,dim=1,keepdim=True))
q_z = VonMisesFisher(mu,  torch.full((1, 1), 50))
b=q_z.rsample(n).view(n,-1)

mu2 = torch.Tensor([[0,-0.5,0.6]])
mu2 = mu2/torch.sqrt(torch.sum(mu2**2,dim=1,keepdim=True))
q_z2 = VonMisesFisher(mu2,  torch.full((1, 1), 50))
c=q_z2.rsample(n).view(n,-1)
# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
mu3 = torch.Tensor([[0,0.2,0]])
mu3 = mu3/torch.sqrt(torch.sum(mu3**2,dim=1,keepdim=True))
q_z3 = VonMisesFisher(mu3,  torch.full((1, 1), 50))
d=q_z3.rsample(n).view(n,-1)


mu4 = torch.Tensor([[0.1,-0.2,-0.1]])
mu4 = mu4/torch.sqrt(torch.sum(mu4**2,dim=1,keepdim=True))
q_z4 = VonMisesFisher(mu4,  torch.full((1, 1), 50))
e=q_z4.rsample(n).view(n,-1)


mu5 = torch.Tensor([[-1,-1,0]])
mu5 = mu5/torch.sqrt(torch.sum(mu5**2,dim=1,keepdim=True))
q_z5 = VonMisesFisher(mu5,  torch.full((1, 1), 50))
f=q_z5.rsample(n).view(n,-1)
# ax.scatter(a[:,0], a[:,1], a[:,2], marker='o')

mu6 = torch.Tensor([[0.1,-0.4,0.6],[-0.1,0.4,-0.6]])
mu6 = mu6/torch.sqrt(torch.sum(mu6**2,dim=1,keepdim=True))
q_z6 = VonMisesFisher(mu6,  torch.tensor([[50],[10]]))
g=q_z6.rsample(n).view(-1,mu6.shape[1])
print(g.shape)

mu = torch.Tensor([[0,1,-1]])
mu = mu/torch.sqrt(torch.sum(mu**2,dim=1,keepdim=True))
q_z = VonMisesFisher(mu,  torch.full((1, 1), 0.001))
uniform=torch.randn((1000,3))
uniform= uniform/torch.sqrt(torch.sum(uniform**2,dim=1,keepdim=True))
# ax.scatter(uniform[:,0], uniform[:,1], uniform[:,2], marker='o',alpha=0.8)
# ax.scatter(b[:,0], b[:,1], b[:,2], marker='1',label='vMF 1',alpha=0.4)
# ax.scatter(c[:,0], c[:,1], c[:,2], marker='^',label='vMF 2',alpha=0.6)
# ax.scatter(d[:,0], d[:,1], d[:,2], marker='+',label='vMF 3',alpha=0.8)
# ax.scatter(b[:,0], b[:,1], b[:,2], marker='^')


##MSSFG
ax.scatter(uniform[:,0], uniform[:,1], uniform[:,2], marker='o',alpha=0.1)
# ax.scatter(b[:,0], b[:,1], b[:,2], marker='1',label='vMF 1',alpha=0.5)
# ax.scatter(c[:,0], c[:,1], c[:,2], marker='^',label='vMF 2',alpha=0.5)
# ax.scatter(d[:,0], d[:,1], d[:,2], marker='+',label='vMF 3',alpha=0.5)
# ax.scatter(e[:,0], e[:,1], e[:,2], marker='2',label='vMF 4',alpha=0.5)
# ax.scatter(f[:,0], f[:,1], f[:,2], marker='P',label='vMF 5',alpha=0.5)
ax.scatter(g[:,0], g[:,1], g[:,2], marker='3',alpha=0.5,color='tab:red')
ax.plot([0,0],[0,0],[0,0])
# ax.plot([0,mu[0,0]],[0,mu[0,1]],[0,mu[0,2]])
# ax.plot([0,mu2[0,0]],[0,mu2[0,1]],[0,mu2[0,2]])
# ax.plot([0,mu3[0,0]],[0,mu3[0,1]],[0,mu3[0,2]])
# ax.plot([0,mu4[0,0]],[0,mu4[0,1]],[0,mu4[0,2]])
# ax.plot([0,mu5[0,0]],[0,mu5[0,1]],[0,mu5[0,2]])
ax.plot([0,mu6[0,0]],[0,mu6[0,1]],[0,mu6[0,2]],color='tab:blue')
ax.plot([0,mu6[1,0]],[0,mu6[1,1]],[0,mu6[1,2]],color='tab:green')


ax.legend()
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('X3')
ax.set_title('Von Mises-Fisher distribution',fontsize=18)
plt.tight_layout()
plt.show()