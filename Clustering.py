#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import math

import pandas as pd
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


def squarify(arr):
        
    l = 0    
    
    if len(arr) == 785:
        l = arr.pop(-1)
    
    n = np.array(arr, dtype=float)
    
    dim = int(math.sqrt(len(arr)))
    
    #add a dimension; 3d is expected for convolution
    m=[n.reshape(dim, dim)]
    
    return np.array(m, dtype=float), l


# In[4]:


class MNISTDataset(Dataset):
    
    def __init__(self, file, train):
        self.train = train
        self.file = file
        self.l = 0
        #print(self.file)
        
    def __len__(self):
        if self.l == 0: 
            f=open(self.file)
            c=csv.reader(f)
            self.l = sum(1 for row in c)
        
        return self.l
    
    def __getitem__(self, idx):
        f=open(self.file)
        c = csv.reader(f)
        
        x=[]
        i=0
        for row in c:
            if i==idx:
                x=row
                break
            i+=1
        f.close()
        
        x,y = squarify(x)
        
        return (torch.from_numpy(x).float(), y)
        
    def test():
        print(self.file)


# In[5]:


train = "mnistData-class/training_x_1k.csv"
val = "mnistData-class/val_data.csv"
test = "mnistData-class/test_x.csv"
train_set = MNISTDataset(file=train, train=True)
val_set = MNISTDataset(file=val, train=True)
test_set = MNISTDataset(file=test, train=True)


# In[6]:


num_workers=0
batch_size=20

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)


# In[7]:


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        #in channels 3, out 32
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        
        self.pool = nn.MaxPool2d(2,2, return_indices=True,padding=0)
        self.depool = nn.MaxUnpool2d(2,2)
        
        self.i=[0,0]
        
        #decoder, a kernel of 2 and stride of 2 increases statial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(16, 6, kernel_size=5, stride=1)
        self.t_conv2 = nn.ConvTranspose2d(6, 1, kernel_size=5, stride=1)
        
    def forward(self, x):
        #encode
        size, x = self.encode(x)
        
        #print(size)
        #print(x.size())
        
        #decode
        x = x.reshape(size)
        
        #x = self.depool(F.relu(self.t_conv1(x)))
        #print(self.i[1])
        x=self.depool(x, self.i[1], output_size=[8,8])
        x=F.relu(x)
        x=self.t_conv1(x)
        #print(x.size())
        
        
        #x = self.depool(F.relu(self.t_conv2(x)), self.i[0])
        #print(x.size())
        #print(self.i[0].size())
        x=self.depool(x, self.i[0], output_size=[24,24])
        x=F.relu(x)
        x=self.t_conv2(x)
        
        #print(x.size())
        
        return x
    
    '''
       encode
       takes the image and returns 1d feature space and size prior to flattening
    '''
    def encode(self, x):
        x  = F.relu(self.conv1(x))
        #print(x.size())
        x, self.i[0] = self.pool(x)
        #print(x.size())

        #x, self.i[1] = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv2(x))
        #print(x.size())
        x, self.i[1] = self.pool(x)
        #print(x.size())
        
        size = x.size()
        
        x2 = []
        for instance in x:
            tmp = torch.flatten(instance)
            #tmp = self.fc(tmp)
            x2.append(tmp)
        
        x3 = torch.ones(batch_size, 256)
        #print(x3)
        for i,t in enumerate(x2):
            x3[i] = t
        #print(x3)
        #print(x3.size())
        
        return size,torch.tensor(x3)


# In[8]:


model = ConvAutoencoder()
print(model)


# In[9]:


criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[10]:


n_epochs = 10

for epoch in range(0, n_epochs):
    train_loss = 0.0
    
    x=1
    tot=50000
    for data in train_loader:
        images, labels = data
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, images)
        
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()*images.size(0)
        
        #print('\r', x,'/50000', sep='', end='', flush=True)
        x+=1
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


# In[11]:


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.squeeze(np.transpose(img, (1, 2, 0))) )


# In[12]:


dataiter = iter(val_loader)
#dataiter.next()


# In[13]:


# obtain one batch of test images
dataiter = iter(val_loader)
images, labels = dataiter.next()

# get sample outputs
output = model(images)
# prep images for display
images = images.numpy()


output = output.view(batch_size, 1, 28, 28)
output = output.detach().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(output[idx])
    
# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])


# In[101]:


encoded = []
realLabels = []
for data in val_loader:
    image, label = data
    
    x = model.encode(image)[1]
    #print(x)
    
    tmp = []
    for row in x:
        encoded.append(row.tolist())
    
    for lbl in label:
        realLabels.append(int(lbl))


# In[15]:


len(encoded[0])


# In[16]:


kmeans = KMeans(n_clusters=10) # You want cluster the passenger records into 2: Survived or Not survived
kmeans.fit(encoded)


# In[17]:


labels = kmeans.labels_


# In[18]:


groups = [] #indices for each cluster
for i in range(10):
    groups.append([])
    
for i,l in enumerate(labels):
    groups[l].append(i)

#
# groups stores the indices/image that are in each cluster
#


# In[19]:


#
# groupLabels is the ground truth that corresponds to each image in a cluster
#

groupLabels = []
for i in range(10):
    groupLabels.append([])
    
for g,group in enumerate(groups):
    for i in group:
        groupLabels[g].append(int(realLabels[i]))


# In[20]:


#
# distro is the amount of each label in each cluster
#

distro = []
for i in range(10):
    distro.append([0,0,0,0,0,0,0,0,0,0])
    
for g,group in enumerate(groupLabels):
    print(g)
    for l in group:
        distro[g][l] = distro[g][l]+1


# In[27]:


distro


# In[50]:


def distroAcc(distro):
    wrong = 0
    total = 0
    for cluster in distro:
        clusterMax=0
    for i,l in enumerate(cluster):
        if l > cluster[clusterMax]:
            clusterMax = i
    
    for i,l in enumerate(cluster):
        total += l
        if i != clusterMax:
            wrong += l

    return(total-wrong)/total


# In[51]:


distroAcc(distro)


# In[32]:


tsne = TSNE(n_components=2, verbose=1, perplexity=25, n_iter=5000)
tsne_results = tsne.fit_transform(encoded)


# In[33]:


tsneData = [[],[]]
tsneData[0] = tsne_results[:,0]
tsneData[1] = tsne_results[:,1]
plt.figure(figsize=(16,10))
plt.scatter(tsneData[0], tsneData[1])
colors = {0:'b', 1:'m', 2:'r', 3:'c', 4:'y', 5:'k', 6:'brown', 7:'burlywood', 8:'orange', 9:'pink'}
for i,pt in enumerate(tsneData[0]):
    plt.scatter(tsneData[0][i], tsneData[1][i], color=colors[int(realLabels[i])])


# In[80]:


kmeans = KMeans(n_clusters=10, max_iter=1000) # You want cluster the passenger records into 2: Survived or Not survived
kmeans.fit(tsne_results)


# In[81]:


labels = kmeans.labels_


# In[82]:


groups = [] #indices for each cluster
for i in range(10):
    groups.append([])
    
for i,l in enumerate(labels):
    groups[l].append(i)
    
groupLabels = []
for i in range(10):
    groupLabels.append([])
    
for g,group in enumerate(groups):
    for i in group:
        groupLabels[g].append(int(realLabels[i]))
        
distro = []
for i in range(10):
    distro.append([0,0,0,0,0,0,0,0,0,0])
    
for g,group in enumerate(groupLabels):
    for l in group:
        distro[g][l] = distro[g][l]+1


# In[83]:


distro


# In[84]:


distroAcc(distro)


# In[110]:


clusterCenters = kmeans.cluster_centers_


# In[111]:


clusterCenters


# In[96]:


plt.figure(figsize=(16,10))
colors = {0:'b', 1:'m', 2:'r', 3:'c', 4:'y', 5:'k', 6:'brown', 7:'burlywood', 8:'orange', 9:'pink'}
for i,pt in enumerate(tsneData[0]):
    plt.scatter(tsneData[0][i], tsneData[1][i], color=colors[int(realLabels[i])])
for i,pt in enumerate(valCenters):
    plt.scatter(pt[0], pt[1], color=colors[i])


# In[102]:


valCentersX = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
valCentersY = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
labelCts = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
for i,lbl in enumerate(realLabels):
    valCentersX[lbl] += tsneData[0][i]
    valCentersY[lbl] += tsneData[1][i]
    labelCts[lbl] += 1
    
for i in range(10):
    valCentersX[i] /= labelCts[i]
    valCentersY[i] /= labelCts[i]


# In[107]:


valCentersX, valCentersY


# In[104]:


def euclideanDist(a, b):
    return math.sqrt( (a[0] - b[0])**2 + (a[1] - b[1])**2 )


# In[123]:


remainingLabels = [0,1,2,3,4,5,6,7,8,9]
dictionary = {}
for i in range(10): #for each clusterCenter
    closestVal = 9999
    swapTo = -1
    
    for lblI,lbl in enumerate(remainingLabels): #look at all unmatched val clusters
        if euclideanDist(valCenters[lbl], [clusterCentersX[i], valCentersY[i]]) < closestVal:
            closestVal = euclideanDist(clusterCenters[lbl], [valCentersX[i], valCentersY[i]])
            swapTo = lbl
    dictionary[i] = swapTo    
    remainingLabels.remove(swapTo)
    print(remainingLabels)


# In[124]:


dictionary


# In[ ]:





# In[32]:


encoded = []
for data in test_loader:
    image, _ = data
    
    x = model.encode(image)[1]
    #print(x)
    
    tmp = []
    for row in x:
        encoded.append(row.tolist())


# In[33]:


kmeans = KMeans(n_clusters=10) # You want cluster the passenger records into 2: Survived or Not survived
kmeans.fit(encoded)
labels = kmeans.labels_
labels


# In[34]:


df = pd.DataFrame(labels, columns=['MNIST_y'])
df.index += 1


# In[35]:


df.to_csv('pred.csv')


# In[ ]:





# In[ ]:


tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)
tsne_results = tsne.fit_transform(encoded)


# In[ ]:


tsneData = [[],[]]
tsneData[0] = tsne_results[:,0]
tsneData[1] = tsne_results[:,1]
plt.figure(figsize=(16,10))
plt.scatter(tsneData[0], tsneData[1])


# In[38]:


kmeans.fit(tsne_results)
labels = kmeans.labels_


# In[41]:


df = pd.DataFrame(labels, columns=['MNIST_y'])
df.index += 1
df.to_csv('pred-tsne.csv')


# In[ ]:


clusterCenters = kmeans.cluster_centers_


# In[ ]:


valCentersX = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
valCentersY = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
labelCts = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
for i,lbl in enumerate(realLabels):
    valCentersX[lbl] += tsneData[0][i]
    valCentersY[lbl] += tsneData[1][i]
    labelCts[lbl] += 1
    
for i in range(10):
    valCentersX[i] /= labelCts[i]
    valCentersY[i] /= labelCts[i]


# In[ ]:


remainingLabels = [0,1,2,3,4,5,6,7,8,9]
dictionary = {}
for i in range(10): #for each clusterCenter
    closestVal = 9999
    swapTo = -1
    
    for lblI,lbl in enumerate(remainingLabels): #look at all unmatched val clusters
        if euclideanDist(valCenters[lbl], [clusterCentersX[i], valCentersY[i]]) < closestVal:
            closestVal = euclideanDist(clusterCenters[lbl], [valCentersX[i], valCentersY[i]])
            swapTo = lbl
    dictionary[i] = swapTo    
    remainingLabels.remove(swapTo)
    print(remainingLabels)


# In[ ]:


redo = []
for i in labels:
    redo.append(dictionary[i])
df = pd.DataFrame(redo, columns=['MNIST_y'])
df.index += 1
df.to_csv('pred_tsne_Rectified.csv')


# In[ ]:





# In[ ]:


torch.save(model.state_dict(), 'weights_only.pth')

