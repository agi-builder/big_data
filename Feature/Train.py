import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import gc
from tqdm import tqdm

from inception_resnet_v1 import *


class DNNTrain(object):
    def __init__(self, network, learning_rate):
        self.network = network
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()


    def train(self, loader, num_epoch):
        self.network.train()
        epoch = 0
        last_loss = 100
        for _ in range(num_epoch):
            epoch +=1
            print('epoch:', epoch)
            gc.collect()
            self.train_epoch(loader['train'], loader['validation'])
    
    def train_epoch(self, train_loader, valid_loader):
        total_loss = 0.0
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = Variable(images)
            labels = Variable(labels)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
                self.network.cuda()
            self.optimizer.zero_grad()
            predictions = self.network(images)
            loss = self.criterion(predictions, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        print('Train loss=', total_loss/i)
            
        valid_loss = 0.0
        for i, (images, labels) in enumerate(valid_loader):
            images = Variable(images)
            labels = Variable(labels)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
                self.network.cuda()
            predictions = self.network(images)
            loss = self.criterion(predictions, labels)
            valid_loss += loss.item()
            break
        
        
#         valid_loss /= i
        acc = get_acc(self.network, valid_loader)
        print('Validation accuracy = ', acc, 'Validation loss = ', valid_loss)




path = './Images'
transform = transforms.Compose([transforms.Resize(160), transforms.ToTensor()])
data_image = {x:datasets.ImageFolder(root = os.path.join(path,x), transform = transform) for x in ['train', 'test']}
index = list(range(len(data_image['train'])))

                    
train_loader = torch.utils.data.DataLoader(data_image['train'], batch_size=100, sampler=SubsetRandomSampler(index[2000:]))
valid_loader = torch.utils.data.DataLoader(data_image['train'], batch_size=100, sampler=SubsetRandomSampler(index[:2000]))
test_loader = torch.utils.data.DataLoader(data_image['test'], batch_size=100, shuffle=True)

data_loader = {'train': train_loader, 'validation': valid_loader, 'test': train_loader}

model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=7, dropout_prob=0.6)
print(model)

trainer = DNNTrain(model, 1e-4)
trainer.train(data_loader,3)