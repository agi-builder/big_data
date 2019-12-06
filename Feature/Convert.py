
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import gc
from tqdm import tqdm
import cv2

from inception_resnet_v1 import *
from Utils import *


model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=4, dropout_prob=0.6)
model = torch.load('./SavedModel/test.pth')
torch.save(model.state_dict(), './SavedModel/dict.pth')