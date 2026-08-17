# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET

Image classification is a fundamental task in computer vision where an input image is assigned to one of several predefined classes. The objective of this experiment is to build and train a Convolutional Neural Network (CNN) using a labeled image dataset and evaluate its performance using accuracy, confusion matrix, and classification report.

## Neural Network Model

<img width="998" height="698" alt="image" src="https://github.com/user-attachments/assets/b9ff98a9-98cc-4632-8dab-20796735f84b" />

## DESIGN STEPS
### STEP 1: 

Import the required libraries and configure the execution device (CPU/GPU) for building the CNN model.

### STEP 2: 

Load the Fashion MNIST dataset, apply image preprocessing, and create training and testing data loaders.

### STEP 3: 

Design and initialize the Convolutional Neural Network (CNN) architecture with convolution, pooling, and fully connected layers.

### STEP 4: 

Define the loss function and optimizer, then train the CNN model using the training dataset.

### STEP 5: 

Evaluate the trained model using the testing dataset and calculate accuracy, confusion matrix, and classification report.

### STEP 6: 

Test the model with a new image and display the actual and predicted class labels


## PROGRAM

### Name:Rohith S

### Register Number:212225240122

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

transform=transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.5,),(0.5))  
])

train_set=torchvision.datasets.FashionMNIST(root='./data',train=True,download=True,
transform=transform)
test_set=torchvision.datasets.FashionMNIST(root='./data',train=False,download=True,
transform=transform)

im,lbl=train_set[0]
print(im.shape)
print(len(train_set))
print(len(test_set))

trl=DataLoader(train_set,batch_size=32,shuffle=True)
tstl=DataLoader(test_set,batch_size=32,shuffle=False)

class CNNclassifier1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.c2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.c3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.l1=nn.Linear(128*3*3,64)
        self.l2=nn.Linear(64,32)
        self.l3=nn.Linear(32,10)
    def forward(self,x):
        x=self.pool(torch.relu(self.c1(x)))
        x=self.pool(torch.relu(self.c2(x)))
        x=self.pool(torch.relu(self.c3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.l1(x))
        x=torch.relu(self.l2(x))
        x=self.l3(x)
        return x

from torchsummary import summary

model=CNNclassifier1()
criterion=nn.CrossEntropyLoss()
op=optim.Adam(model.parameters(),lr=0.001)
if torch.cuda.is_available():
    print(torch.cuda.is_available())
    device=torch.device('cuda')
    model.to(device)


summary(model,input_size=(1,28,28))


epochs=3
running_loss=0.0

for i in range(epochs):
    model.train()
    for a,b in trl:
        op.zero_grad()
        pred=model(a)
        loss=criterion(pred,b)
        loss.backward()
        op.step()
        running_loss+=loss.item()
    print(f"Loss:{i}",running_loss/len(trl))

t=0
c=0
act=[]
pre=[]
model.eval()
with torch.no_grad():
    for img,labels in tstl:
        output=model(img)
        _,predicted=torch.max(output,1)
        t=t+labels.size(0)
        c+=(predicted==labels).sum().item()
        pre.extend(predicted.cpu().numpy())
        act.extend(labels.cpu().numpy())
accuracy=c/t*100
print("Accuracy Score:",accuracy)
conf_matrix=confusion_matrix(act,pre)
class_report=classification_report(act,pre,target_names=test_set.classes)
print("Classification Report:",class_report)
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues',xticklabels=test_set.classes,yticklabels=test_set.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

with torch.no_grad():
    img1,label=test_set[0]
    output=model(img1.unsqueeze(0))
    _,pred=torch.max(output,1)
    classes=test_set.classes
    img1 = img1 * 0.5 + 0.5
    plt.imshow(img1.squeeze(),cmap="gray")
    plt.title("Predicted Image")
    plt.axis('off')
    plt.show()
    print(f"Actual:{classes[label]}\n")
    print(f"Predicted:{classes[pred.item()]}\n")


original_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=None
)

image, label = original_dataset[0]

plt.imshow(image, cmap='gray')
plt.title(original_dataset.classes[label])
plt.axis('off')
plt.show()


```

### OUTPUT

## Training Loss per Epoch

<img width="407" height="101" alt="image" src="https://github.com/user-attachments/assets/4a5a1f22-b26f-4b86-b957-29741ed5c1e2" />

## Confusion Matrix

<img width="1002" height="682" alt="image" src="https://github.com/user-attachments/assets/8e46bc75-1082-4bb3-b189-14faf1ef8910" />

## Classification Report

<img width="844" height="406" alt="image" src="https://github.com/user-attachments/assets/857203e8-6a8e-4df7-a249-0e33c1c1d1f4" />

### New Sample Data Prediction

<img width="728" height="678" alt="image" src="https://github.com/user-attachments/assets/d591cfe0-1139-4aaa-b36f-65cfe513fd64" />

## RESULT

The CNN model was successfully trained and tested on the Fashion MNIST dataset, achieving accurate image classification with performance evaluation metrics.
