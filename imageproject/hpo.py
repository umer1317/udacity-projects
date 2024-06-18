#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import logging
import sys
import os
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json

    
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 133)
    
    return model


def train(model, train_loaders, epochs, criterion, optimizer, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    for epoch in range(epochs):
        for phase in ['train','valid']:
            running_loss = 0
            running_correct = 0
            
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            for data,target in train_loaders[phase]:
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                loss = criterion(outputs,target)
                _,preds = torch.max(outputs,1)
                running_loss += loss.item() * data.size(0)
                
                with torch.no_grad():
                    running_correct += torch.sum(preds == target).item()
                    
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
            epoch_loss = running_loss / len(train_loaders[phase].dataset)
            epoch_accuracy = running_correct/len(train_loaders[phase].dataset)

            print(f'Epoch : {epoch}-{phase}, epoch loss = {epoch_loss}, epoch_acc = {epoch_accuracy}')
    return model

def model_fn(model_dir):
    print("In model_function. Model directory is -")
    print(model_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = net().to(device)

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        print("Loading the dog classifier model")
        checkpoint = torch.load(f , map_location =device)
        model.load_state_dict(checkpoint)
        print('MODEL-LOADED')
        logger.info('model loaded successfully')
    model.eval()
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''


    traindata = os.path.join(data, "train")
    validdata = os.path.join(data, "valid")
    testdata = os.path.join(data, "test")
    
    
    transform = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406),
                                                          (0.229, 0.224, 0.225))
                                   ])
    
    trainset = torchvision.datasets.ImageFolder(traindata, transform=transform)
    validset = torchvision.datasets.ImageFolder(validdata, transform=transform)
    testset = torchvision.datasets.ImageFolder(testdata, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    
    return trainloader, validloader, testloader

def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    model.eval()
    running_loss = 0
    running_corrects = 0
    
    for inputs,labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _,preds = torch.max(outputs,1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        
    total_loss = running_loss/len(test_loader.dataset)
    total_accuracy = running_corrects/len(test_loader.dataset)
    print(f"Testing Accuracy: {100*total_accuracy}, Testing Loss: {total_loss}")

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    
    
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.Adam(model.fc.parameters(),args.lr)
    
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
        
    trainloader, validloader, testloader = create_data_loaders(args.data_dir, args.batch_size)
    train_loader = trainloader
    valid_loader = validloader
    test_loader = testloader
    
    train_loaders = {
        'train' : train_loader,
        'valid' : valid_loader
    }

    model=train(model, train_loaders, args.epochs, loss_criterion, optimizer, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="E",
        help="number of epochs to train (default: 1)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    # learning rate
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model-dir',type=str,default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-dir',type=str,default=os.environ['SM_OUTPUT_DATA_DIR'])
   
    args=parser.parse_args()
    main(args)
