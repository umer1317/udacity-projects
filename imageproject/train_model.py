#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import argparse
import logging
import os
import time
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


import torch
import torch.nn as nn

import torch.optim as optim

try:
    import smdebug.pytorch as smd
except:
    pass


#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader, criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    hook.set_mode(smd.modes.EVAL)
    model.eval()
    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)

    print(f"Testing Loss: {total_loss}")
    print(f"Testing Accuracy: {total_acc}")


def train(model,train_loader,validation_loader,criterion,optimizer,device,hook,early_stopping):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    epochs = 2
    best_loss = 1e6
    image_dataset = {'train': train_loader, 'valid': validation_loader}
    loss_counter = 0
    hook.set_mode(smd.modes.TRAIN)

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples = 0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples += len(inputs)
                if running_samples % 2000 == 0:
                    accuracy = running_corrects / running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0 * accuracy,
                        )
                    )

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples

            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1
            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase,
                                                                                        epoch_loss,
                                                                                        epoch_acc,
                                                                                        best_loss))
        if loss_counter == early_stopping:
            break
    return model

    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained = True)
    
    for param in model.parameters():
        param.require_grad = False
        
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 133)
    
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    dataloaders = {
        split : torch.utils.data.DataLoader(data[split], batch_size, shuffle=True)
        for split in ['train', 'valid', 'test']
    }

    return dataloaders

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(),args.lr)
    hook.register_loss(loss_criterion)
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    data_transforms = {
        'train' : transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ]),
        'valid' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ]),
        'test' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
    }
        
    image_datasets = {
        split : datasets.ImageFolder(os.path.join(args.data_dir, split) , data_transforms[split])
        for split in ['train','test','valid']
    }
    
    dataloaders = create_data_loaders(image_datasets, args.batch_size)
    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']
    test_loader = dataloaders['test']
    
    train_loaders = {
        'train' : train_loader,
        'valid' : valid_loader
    }
    
    model = train(model,
                  train_loader,
                  valid_loader,
                  loss_criterion,
                  optimizer,
                  device,
                  hook,
                  early_stopping=args.early_stopping_rounds)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device, hook)
    
    '''
    TODO: Save the trained model
    '''

    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="E",
        help="number of epochs to train (default: 2)",
    )
    # batch_size
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument('--early-stopping-rounds',
                        type=int,
                        default=10)
    # learning rate
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 1.0)"
    )
    
    parser.add_argument('--data-dir', type=str,default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model-dir',type=str,default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-dir',type=str,default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)
