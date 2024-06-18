import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'
ACCEPTED_CONTENT_TYPE = [ JPEG_CONTENT_TYPE ]





def model_fn(model_dir):
    print("In model_fn. Model directory is -")
    print(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        print("Loading the dog-classifier model")
        checkpoint = torch.load(f , map_location =device)
        model.load_state_dict(checkpoint, strict=False)
        print('MODEL-LOADED')
        print('model loaded successfully')
    model.eval()
    return model






        
def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features

    model.fc = nn.Sequential(
                   nn.Linear(num_features, 256),
                   nn.ReLU(inplace=True),
                   nn.Linear(256, 133))
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

def predict_fn(input_object, model):
    print('In predict fn')
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])
    print("transforming input")
    input_object=test_transform(input_object)
    
    with torch.no_grad():
        print("Calling model")
        prediction = model(input_object.unsqueeze(0))
    return prediction

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    print('Deserializing the input data.')
    
    print(f'Incoming Requests Content-Type is: {content_type}')
    print(f'Request body Type is: {type(request_body)}')
    if content_type in ACCEPTED_CONTENT_TYPE:
        print(f"Returning an image of type {content_type}" )
        return Image.open(io.BytesIO(request_body))
    else:
        raise Exception(f"Requested an unsupported Content-Type: {content_type}, Accepted Content-Type are: {ACCEPTED_CONTENT_TYPE}")