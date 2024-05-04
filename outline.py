# Import packages 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import gradio as gr

# Transform data and load dataset

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#Upload and unify dataset
base_dir = 'kagglecatsanddogs_3367a/PetImages'

#Create a unified dataset
dataset = datasets.ImageFolder(root=base_dir, transform=transform)

#Training and validation sets
train_set, val_set = train_test_split(dataset, test_size=0.2, random_state=42)

#Create data and validation loaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

#Create CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        #Convonlutional layer
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  #takes 3-channel image -> output 16 feature maps
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) #takes 16-channel image -> output 32 feature maps
        self.pool = nn.MaxPool2d(2, 2) #Pool layer
        #Flatten and output layers
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        #Apply conv layer. Then relu and max pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()

#Train model
criterion = nn.CrossEntropyLoss() #Loss function setup
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #Setup SGD

num_epochs = 5 #number of dataset iterations
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad() #Clear gradient for training
        outputs = model(images)

        #Calculate and summ losses
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}") #Print average epoch loss

#Variable tracking
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images) #compute predictions
        _, predicted = torch.max(outputs.data, 1) #Get predicted classes
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Validation accuracy: {100 * correct / total}%') #Print validation set accuracy


def predict(image_path, model):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Make a prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return 'cat' if predicted.item() == 0 else 'dog'

#### GRADIO UI #####
def predict(inp):
    image_transform = transforms.Compose([
        transforms.Resize((64, 64)),  #Change size for training and keep consistent
        transforms.ToTensor(),  #Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  #Normalize
    ])

    #Apply transformations to the input image
    inp = image_transform(inp).unsqueeze(0)

    labels = ['cat', 'dog']

    with torch.no_grad():
        prediction = model(inp) 
        softmax = torch.nn.functional.softmax(prediction, dim=1)
    
        #Confidence for each label
        confidences = {labels[i]: float(softmax.squeeze()[i]) for i in range(len(labels))}    

    return confidences

#Gradio interface
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Cat vs. Dog Classifier",
    description="Upload an image of a cat or a dog, and this classifier will predict which it is.",
).launch()