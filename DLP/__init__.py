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


########################################################################


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

    return z    
    
    
    
    
    
    
    
