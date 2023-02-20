#!/usr/bin/env python

"""

"""
from PIL import Image
import pandas as pd
import glob
import gc
import os 
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import models, transforms
import imgaug
from imgaug import augmenters as iaa
import imgaug as ia
import matplotlib.pyplot
from datetime import datetime
import time
import json
import scipy.stats
import munch
import argparse

#python functions/PentaMind.py --model_name vgg19_bn --use_pretrained True --pheno_file_dir data/phenotype/ --image_file_dir data/images/ --output_dir results/PentaMind/ --optim sgd --batch-size 32 --epochs 90 --lr 0.001 --momentum 0.9 --weight-decay 0.0001
# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--model_name', type=str, required=True, help="Name of the model implemented in torchvision")
parser.add_argument('--use_pretrained', type=str, required=True, help="Use pretrained weights")
parser.add_argument('--pheno_file_dir', type=str, required=True, help="Location of the directory containing phenotype files")
parser.add_argument('--image_file_dir', type=str, required=True, help="Location of the directory containing image files")
parser.add_argument('--output_dir', type=str, required=True, help="Location of the directory for output")
parser.add_argument('--optim', default='sgd', type=str, required=True, help="Optimizer")
parser.add_argument("--batch-size", default=32, type=int, help="Images per GPU, the total batch size is NGPU x batch_size")
parser.add_argument("--epochs", default=90, type=int, metavar="N", help="Number of total epochs to run")
parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="Momentum")
parser.add_argument("--weight-decay",default=1e-4,type=float,metavar="W",help="weight decay (default: 1e-4)",dest="weight_decay")

# Parse the argument
args = parser.parse_args()
config = {
    'project': "PentaMind",
    'batch_size': args.batch_size,
    'max_epochs': args.epochs,
    'model_name': args.model_name, 
    'feature_extract': False, 
    'use_pretrained': args.use_pretrained=="True",
    'pheno_file_dir': args.pheno_file_dir,
    'image_file_dir': args.image_file_dir,
    'output_dir': args.output_dir,
    'opt': args.optim,
    'Fliplr':True,
    'Flipud':True,
    'Rot90':True,
    'Affine':True,
    'aff_sometimes':0.9,
    'aff_translate':0.1,
    'aff_rotate':10,
    'CBS':True,
    'AdditiveGaussianNoise':False,
    'SaltAndPepper':False,
    'lr': args.lr,
    'momentum': args.momentum,
    'weight_decay': args.weight_decay
}
config=munch.Munch(config)

class ImgAugTransform:
    def __init__(self,params):
        aug_list = []
        if params['Fliplr']:
            aug_list.append(iaa.Fliplr(0.5))
        if params['Flipud']:
            aug_list.append(iaa.Flipud(0.5))
        if params['Rot90']:
            aug_list.append(iaa.Rot90((0,3)))
        if params['Affine']:
            aug_list.append(iaa.Sometimes(params['aff_sometimes'],
                                          iaa.Affine(translate_percent={'x':(-params['aff_translate'],
                                                                             params['aff_translate']),
                                                                        'y':(-params['aff_translate'],
                                                                             params['aff_translate'])},
                                                     rotate=(-params['aff_rotate'], params['aff_rotate']),
                                                     cval=255)))
        aug_list2 = []
        if params['CBS']:
            aug_list2.append(iaa.SomeOf((0, 3), [
                iaa.GammaContrast((0, 2.0)),
                iaa.GaussianBlur(sigma=(0, 3.0)),
                iaa.Sharpen((0, 1))
            ],random_order=True))
        if params['AdditiveGaussianNoise']:
            aug_list2.append(iaa.AdditiveGaussianNoise(scale=(0, 0.02*255)))
        if params['SaltAndPepper']:
            aug_list2.append(iaa.SaltAndPepper(0.05))
        
        self.aug = iaa.Sequential(aug_list, 
                                  random_order=True)
        self.aug2 = iaa.Sequential(aug_list2, 
                                  random_order=True)
    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        img = self.aug2.augment_image(img)
        return img


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self,params,set_name, aug):
        'Initialization'
        pheno=pd.read_csv(params["pheno_file_dir"]+set_name,sep="\t")
        self.image_file_dir = params["image_file_dir"]
        self.pheno = torch.from_numpy(np.array(pheno.values[:,1]).astype(np.float64))
        self.file_name = np.array(pheno.values[:,0])
        self.transform_img = transforms.Compose([transforms.Resize((params["input_size"],params["input_size"])),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])
        self.aug=aug
        if self.aug:
            self.transforms_imgaug = ImgAugTransform(params)
    def __len__(self):
        'Denotes the total number of samples'
        return self.pheno.shape[0]
    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = Image.open(self.image_file_dir+self.file_name[index], 'r')
        if self.aug:
            X = self.transforms_imgaug(X)
            X = Image.fromarray(X)
        X = self.transform_img(X)
        y = self.pheno[index]
        return X, y
    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def initialize_model(model_name, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    num_classes = 1
    if ('resnet' in model_name) & (not ('wide' in model_name)):
        if model_name == "resnet18":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
        elif model_name == "resnet34":
            """ Resnet34
            """
            model_ft = models.resnet34(pretrained=use_pretrained)
        elif model_name == "resnet50":
            """ Resnet50
            """
            model_ft = models.resnet50(pretrained=use_pretrained)
        elif model_name == "resnet101":
            """ Resnet101
            """
            model_ft = models.resnet101(pretrained=use_pretrained)
        elif model_name == "resnet152":
            """ Resnet152
            """
            model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "googlenet":
        """ googlenet
        """
        model_ft = models.googlenet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif 'wide_resnet' in model_name:
        if model_name == "wide_resnet50_2":
            """ wide_resnet50_2
            """
            model_ft = models.wide_resnet50_2(pretrained=use_pretrained)
        elif model_name == "wide_resnet101_2":
            """ wide_resnet101_2
            """
            model_ft = models.wide_resnet101_2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif 'resnext' in model_name:
        if model_name == "resnext50_32x4d":
            """ resnext50_32x4d
            """
            model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        elif model_name == "resnext101_32x8d":
            """ resnext101_32x8d
            """
            model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif 'mnasnet' in model_name:
        if model_name == "mnasnet0_5":
            """ mnasnet0_5
            """
            model_ft = models.mnasnet0_5(pretrained=use_pretrained)
        elif model_name == "mnasnet0_75":
            """ mnasnet0_75
            """
            model_ft = models.mnasnet0_75(pretrained=use_pretrained)
        elif model_name == "mnasnet1_0":
            """ mnasnet1_0
            """
            model_ft = models.mnasnet1_0(pretrained=use_pretrained)
        elif model_name == "mnasnet1_3":
            """ mnasnet1_3
            """
            model_ft = models.mnasnet1_3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif 'mobilenet_v3' in model_name:
        if model_name == "mobilenet_v3_large":
            """ mobilenet_v3_large
            """
            model_ft = models.mobilenet_v3_large(pretrained=use_pretrained)
        elif model_name == "mobilenet_v3_small":
            """ mobilenet_v3_small
            """
            model_ft = models.mobilenet_v3_small(pretrained=use_pretrained)
            
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[3] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif 'mobilenet_v2' in model_name:
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif 'vgg' in model_name:
        if model_name == "vgg11":
            """ VGG11
            """
            model_ft = models.vgg11(pretrained=use_pretrained)
        elif model_name == "vgg11_bn":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
        elif model_name == "vgg13":
            """ VGG13
            """
            model_ft = models.vgg13(pretrained=use_pretrained)
        elif model_name == "vgg13_bn":
            """ VGG13_bn
            """
            model_ft = models.vgg13_bn(pretrained=use_pretrained)
        elif model_name == "vgg16":
            """ VGG16
            """
            model_ft = models.vgg16(pretrained=use_pretrained)
        elif model_name == "vgg16_bn":
            """ VGG16_bn
            """
            model_ft = models.vgg16_bn(pretrained=use_pretrained)
        elif model_name == "vgg19":
            """ VGG19
            """
            model_ft = models.vgg19(pretrained=use_pretrained)
        elif model_name == "vgg19_bn":
            """ VGG19_bn
            """
            model_ft = models.vgg19_bn(pretrained=use_pretrained)
            
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif 'efficientnet' in model_name:
        if model_name == "efficientnet_b0":
            """ efficientnet_b0
            """
            model_ft = models.efficientnet_b0(pretrained=use_pretrained)
        elif model_name == "efficientnet_b1":
            """ efficientnet_b1
            """
            model_ft = models.efficientnet_b1(pretrained=use_pretrained)
        elif model_name == "efficientnet_b2":
            """ efficientnet_b2
            """
            model_ft = models.efficientnet_b2(pretrained=use_pretrained)
        elif model_name == "efficientnet_b3":
            """ efficientnet_b3
            """
            model_ft = models.efficientnet_b3(pretrained=use_pretrained)
        elif model_name == "efficientnet_b4":
            """ efficientnet_b4
            """
            model_ft = models.efficientnet_b4(pretrained=use_pretrained)
        elif model_name == "efficientnet_b5":
            """ efficientnet_b5
            """
            model_ft = models.efficientnet_b5(pretrained=use_pretrained)
        elif model_name == "efficientnet_b6":
            """ efficientnet_b6
            """
            model_ft = models.efficientnet_b6(pretrained=use_pretrained)
        elif model_name == "efficientnet_b7":
            """ efficientnet_b7
            """
            model_ft = models.efficientnet_b7(pretrained=use_pretrained)
            
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif 'regnet' in model_name:
        if model_name == "regnet_y_400mf":
            """ regnet_y_400mf
            """
            model_ft = models.regnet_y_400mf(pretrained=use_pretrained)
        elif model_name == "regnet_y_800mf":
            """ regnet_y_800mf
            """
            model_ft = models.regnet_y_800mf(pretrained=use_pretrained)
        elif model_name == "regnet_y_1_6gf":
            """ regnet_y_1_6gf
            """
            model_ft = models.regnet_y_1_6gf(pretrained=use_pretrained)
        elif model_name == "regnet_y_3_2gf":
            """ regnet_y_3_2gf
            """
            model_ft = models.regnet_y_3_2gf(pretrained=use_pretrained)
        elif model_name == "regnet_y_8gf":
            """ regnet_y_8gf
            """
            model_ft = models.regnet_y_8gf(pretrained=use_pretrained)
        elif model_name == "regnet_y_16gf":
            """ regnet_y_16gf
            """
            model_ft = models.regnet_y_16gf(pretrained=use_pretrained)
        elif model_name == "regnet_y_32gf":
            """ regnet_y_32gf
            """
            model_ft = models.regnet_y_32gf(pretrained=use_pretrained)
        elif model_name == "regnet_x_400mf":
            """ regnet_x_400mf
            """
            model_ft = models.regnet_x_400mf(pretrained=use_pretrained)
        elif model_name == "regnet_x_800mf":
            """ regnet_x_800mf
            """
            model_ft = models.regnet_x_800mf(pretrained=use_pretrained)
        elif model_name == "regnet_x_1_6gf":
            """ regnet_x_1_6gf
            """
            model_ft = models.regnet_x_1_6gf(pretrained=use_pretrained)
        elif model_name == "regnet_x_3_2gf":
            """ regnet_x_3_2gf
            """
            model_ft = models.regnet_x_3_2gf(pretrained=use_pretrained)
        elif model_name == "regnet_x_8gf":
            """ regnet_x_8gf
            """
            model_ft = models.regnet_x_8gf(pretrained=use_pretrained)
        elif model_name == "regnet_x_16gf":
            """ regnet_x_16gf
            """
            model_ft = models.regnet_x_16gf(pretrained=use_pretrained)
        elif model_name == "regnet_x_32gf":
            """ regnet_x_32gf
            """
            model_ft = models.regnet_x_32gf(pretrained=use_pretrained)
            
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif model_name == "squeezenet1_1":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_1(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224
    elif model_name == "densenet121":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size

def define_network(config):
    net, input_size = initialize_model(config.model_name, config.feature_extract, use_pretrained=config.use_pretrained)
    net.to(config.device)
    #print(net)

    params_to_update = net.parameters()
    if config.feature_extract:
        params_to_update = []
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

    opt_name = config.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            params_to_update,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            params_to_update, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(params_to_update, lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {config.opt}. Only SGD, RMSprop and AdamW are supported.")
        
    def my_loss(output, target):
        loss = torch.mean((output - target)**2)
        return loss
    criterion = my_loss
    
    return net, criterion, optimizer, input_size

def build_model(config):
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config.device.type!="cpu":
        torch.cuda.empty_cache()
    gc.collect()
    if config.device.type!="cpu":
        torch.cuda.empty_cache()
    
    
    
    net, criterion, optimizer, input_size = define_network(config)
    config.input_size = input_size

    # Parameters
    params_dl = {'batch_size': config.batch_size,
                 'shuffle': True,
                 'num_workers': os.cpu_count(),
                 'pin_memory':True}

    # initial loss
    best_loss=float("inf")
    
    # count epoch showing no improvement
    not_improve=0
    patience=3
    
    torch.backends.cudnn.benchmark = True
    
    for epoch in range(config.max_epochs):
        # training
        training_generator = data.DataLoader(Dataset(config,set_name="train.txt",aug=True), **params_dl)
        train_loss=0
        #start_time = time.time()
        for i,(X, y) in enumerate(training_generator):
            X, y = X.to(config.device, non_blocking=True), y.float().to(config.device, non_blocking=True)
            net.train()
            # Model computation
            optimizer.zero_grad()   # zero the gradient buffers
            output = net(X)
            loss = criterion(output[:,0], y)  
            loss.backward()
            optimizer.step()    # Does the update
            train_loss += loss.detach() * len(y) / len(training_generator.dataset)
            
        del training_generator, X, y, loss, output
        torch.cuda.empty_cache()

        # validation 
        valid_loss=0
        valid_generator = data.DataLoader(Dataset(config,set_name="valid.txt",aug=True), **params_dl)
        y_valid=[]
        y_pred=[]
        with torch.no_grad():
            for i,(X, y) in enumerate(valid_generator):
                X, y = X.to(config.device, non_blocking=True), y.float().to(config.device, non_blocking=True)
                # Model computation
                net.eval()
                output = net(X)
                loss = criterion(output[:,0], y)
                valid_loss += loss.detach() * len(y) / len(valid_generator.dataset)
                y_valid.append(np.array(y.cpu().detach()))
                y_pred.append(np.array(output.cpu().detach()[:,0]))
        del valid_generator, X, y, loss, output
        torch.cuda.empty_cache()
        
        valid_cor=scipy.stats.spearmanr(np.concatenate(y_valid),np.concatenate(y_pred))
        print("Epoch: {}, Training Loss: {}, Validation Loss: {}, Validation Correlation: {}".format(epoch+1, round(train_loss.item(),2), round(valid_loss.item(),2),  round(valid_cor[0],2)))        
        # early stopping
        if valid_loss<best_loss:
            best_loss=valid_loss
            not_improve=0
            best_prams = net.state_dict()
        else:
            not_improve=not_improve+1

        if not_improve>=patience: 
            break

    torch.cuda.empty_cache()

    # testing
    # Load best model
    net.load_state_dict(best_prams)
    params_dl = {'batch_size': config.batch_size,
                 'shuffle': False,
                 'num_workers': os.cpu_count(),
                 'pin_memory':True}
    test_generator = data.DataLoader(Dataset(config,set_name="test.txt",aug=False), **params_dl)
    y_test=[]
    y_pred=[]
    with torch.no_grad():
        for i,(X, y) in enumerate(test_generator):
            X = X.to(config.device, non_blocking=True)
            # Model computation
            net.eval()
            output = net(X)
            y_test.append(np.array(y))
            y_pred.append(np.array(output.cpu().detach()[:,0]))
    del X, y, test_generator, output, optimizer, best_prams

    y_test=np.concatenate(y_test)
    y_pred=np.concatenate(y_pred)
    test_cor=scipy.stats.spearmanr(y_test,y_pred)
    test_loss=criterion(torch.tensor(y_pred), torch.tensor(y_test))
    
    print("Test Loss: {}, Test Correlation: {}".format(round(test_loss.item(),2),  round(valid_cor[0],2)))        

    del y_test, criterion
    # clear chache on GPU
    gc.collect()
    if config.device.type!="cpu":
        torch.cuda.empty_cache()
    gc.collect()
    if config.device.type!="cpu":
        torch.cuda.empty_cache()
    
    return net, y_pred
  
def main():
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    net, y_pred =build_model(config)
    torch.save(net.state_dict(), config.output_dir+'model.pth')
    np.save(config.output_dir+"y_pred.npy", y_pred)

if __name__ == '__main__':
    main()
