import torch
import imageio
import requests
import torchvision
import numpy as np
from math import exp
from PIL import Image
from time import time
from scipy import signal
from random import random
from matplotlib import cm
from bisect import bisect
from random import uniform
from random import randrange
from torchvision import models
from skimage import io as skio
import matplotlib.pyplot as plt
from torchvision.models import *
from torchsummary import summary
from skimage.util import montage
from torch.autograd import Variable
from scipy.special import factorial
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from torchvision import models, transforms

import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import pylab
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.utils


from random import random
from random import uniform
from random import randrange
from math import exp
from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LightSource


import matplotlib.pyplot as plt
import numpy as np
from skimage import io as io
from skimage.util import view_as_blocks
from skimage.transform import resize
from scipy import signal
import torch.nn.functional as F
from torch.nn.functional import *
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
from matplotlib import animation, rc
from IPython.display import HTML
rc('animation', html='html5')


import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import pylab
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.utils

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cbook as cbook
import numpy as np
import scipy.ndimage
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter 
# from scipy.misc import imresize
from scipy import signal
from skimage import io
from sklearn.preprocessing import scale
from PIL import Image
import time
import torch
import numpy as np
import torch.nn as nn
from scipy import stats
from skimage import io as io
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage.util import montage
from torch.nn.functional import *
from torch.autograd import Variable
from torchvision import datasets, transforms


import os
import sys
import time
import copy
import json
import torch
import pylab
import torch
import pylab
import random
import subprocess
import torchvision
import numpy as np
import torchvision
import pandas as pd
import torch.nn as nn
import seaborn as sns
import torch.nn as nn
from scipy import stats
import torchvision.utils
from torch import matmul
import torchvision.utils
from bisect import bisect
import torch.optim as optim
from skimage import io as io
from numpy import linalg as LA
from google.colab import drive
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.util import montage
from torch.nn.functional import *
from torchvision import transforms
from random import random, randint
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler
from sklearn.decomposition import PCA
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms
from urllib.request import Request, urlopen
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from sklearn.linear_model import LogisticRegression as LR
from torch.utils.data import DataLoader, TensorDataset, random_split






import numpy as np
import sklearn.preprocessing as skp
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import scipy.io as sio
from skimage.util import montage




subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'cloud-tpu-client==0.10', 'https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.7-cp36-cp36m-linux_x86_64.whl'])    
import torch
import torch_xla
import torch_xla.core.xla_model as xm






subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'wandb'])
import wandb as wb

########################################################################


def plot(x):
    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap = 'gray')
    ax.axis('off')
    fig.set_size_inches(18, 10)
    plt.show()
    
def bar(a):
    x = np.arange(a.shape[0])
    fig, ax = plt.subplots()
    plt.bar(x, a)
    # plt.xticks(x, ('0', '1', '2', '3', '4','5', '6', '7', '8', '9'))
    plt.show()


def GPU(data):
    return torch.tensor(data, requires_grad=True, dtype=torch.float, device=torch.device('cuda'))

def GPU_data(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float, device=torch.device('cuda'))
  
 
def cross_entropy(outputs, labels):            
    return -torch.sum(softmax(outputs).log()[range(outputs.size()[0]), labels.long()])/outputs.size()[0]    


  
def softmax(x):
    s1 = torch.exp(x - torch.max(x,1)[0][:,None])
    s = s1 / s1.sum(1)[:,None]
    return s  


def acc(out,y):
    with torch.no_grad():
        return (torch.sum(torch.max(out,1)[1] == y).item())/y.shape[0]


def plot(x):
    if type(x) == torch.Tensor :
        x = x.cpu().detach().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap = 'gray')
    ax.axis('off')
    fig.set_size_inches(10, 10)
    plt.show()
    
    
    
def montage_plot(x):
    x = np.pad(x, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    plot(montage(x))
    
    
    
    
    
    
    
def Truncated_Normal(size):

    u1 = torch.rand(size)*(1-np.exp(-2)) + np.exp(-2)
    u2 = torch.rand(size)
    z  = torch.sqrt(-2*torch.log(u1)) * torch.cos(2*np.pi*u2)
    z = 0.1*z
    return z    
    
    
    
    
    
    
def load_MNIST(dataset):

    if dataset == "MNIST":
        train_set = datasets.MNIST('./data', train=True, download=True)
        test_set = datasets.MNIST('./data', train=False, download=True)

    if dataset == "KMNIST":
        train_set = datasets.KMNIST('./data', train=True, download=True)
        test_set = datasets.KMNIST('./data', train=False, download=True)

    if dataset == "FMNIST":
        train_set = datasets.FashionMNIST('./data', train=True, download=True)
        test_set = datasets.FashionMNIST('./data', train=False, download=True)


    X = train_set.data.numpy()
    X_test = test_set.data.numpy()
    Y = train_set.targets.numpy()
    Y_test = test_set.targets.numpy()

    X = X[:,None,:,:]/255
    X_test = X_test[:,None,:,:]/255

    return X,Y,X_test,Y_test    
