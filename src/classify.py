import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import ImageFile
from sklearn.metrics import balanced_accuracy_score
    

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_data(model_name, datadir):
    resize_value = 256
    crop_value = 224
    
    if model_name == 'inception':
        resize_value = 310
        crop_value = 299
        
    data_transform = transforms.Compose([
        transforms.Resize(resize_value),
        transforms.CenterCrop(crop_value),
        transforms.ToTensor() ,
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    data_dir = datadir+'/'
    image_dataset = ImageFolderWithPaths(data_dir+'test',
                                          data_transform) 
                  
    batch_size = 16

    
    dataloader =  DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
              
    return dataloader

def get_model(model_name, filename):

    feature_extract = False
    num_classes = 16
    model_ft = None
    if model_name == 'alexnet':
        model_ft = models.alexnet(pretrained=True)
        model_ft.classifier[6] = nn.Linear(4096,num_classes)
    if model_name == 'vgg19_bn':
        model_ft = models.vgg19_bn(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)    
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    if model_name == 'resnet50':
        model_ft = models.resnet50(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)    
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)     
    if model_name == 'resnet152':
        model_ft = models.resnet152(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)    
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)   
    if model_name == 'inception':    
        model_ft = models.inception_v3(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)   
    if model_name == "densenet":
        model_ft = models.densenet201(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)     

    model_ft = model_ft.cuda()
    model_ft.load_state_dict(torch.load(filename))
    return model_ft

def eval_model(model, data):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    res = []
    y_true = []
    y_pred = []
    model.eval()

    with torch.no_grad():
        for i, inputs  in enumerate(data):
            filenames = inputs[2]
            inputs = inputs[0]
            inputs = inputs.cuda()
            

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                value = int(preds[j])
                file_id = int(filenames[j].replace('.jpg','').replace('data/test/0/',''))
                res.append( (file_id,value) )

    return res

def get_prediction(model_name, filename, datadir):
    data = get_data(model_name,datadir)
    model = get_model(model_name, filename)
    res = eval_model(model, data)
    res.sort()
    return res

def save_to_file(filename, prediction):
    f = open(filename, "w")
    for p in prediction:
        f.write(str(p[0])+','+str(p[1])+'\n')
    f.close()


def main():
    parser = argparse.ArgumentParser(description='Predicts and exports the prediction to csv')
    #parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #               help='an integer for the accumulator')
    parser.add_argument('--filename',  help='Indicates state_dict or pre-calculated weights filename', 
        metavar='filename')
    parser.add_argument('--datadir', help='Indicates state_dict or pre-calculated weights filename', 
        metavar='datadir')

    args = parser.parse_args()
    
    filename = args.filename
    datadir = args.datadir
    print(filename)


    model_name = filename.split('_')[0]
    
    print('Nombre del modelo (Transfer Learning):',model_name)

    prediction  = get_prediction(model_name, filename, datadir)

    
    target_filename = filename.replace('pth', 'csv')
    save_to_file(target_filename, prediction)
    print('Las predicciones fueron salvadas en el archivo: '+target_filename)
    

if __name__ == '__main__':
    main()    