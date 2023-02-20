
import torch.optim as optim
import torch.nn as nn
import torch
import copy
import time
import scipy.stats
import pandas as pd
import os 
import numpy as np
import munch
import matplotlib.pyplot
import matplotlib
import json
import imgaug as ia
import imgaug as ia
import imgaug
import h5py
import glob
import gc
import argparse
from torchvision import models, transforms
from torch.utils import data
from simulate_pentagon import simulate_pentagon
from PIL import Image
from imgaug import augmenters as iaa
from datetime import datetime

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
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
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

def define_network(config):
    net, input_size = initialize_model(config.model_name, False, use_pretrained=True)
    net.to(config.device)
    return net, input_size


# image generation
np.random.seed(1234)
param_list = [{'n':5,
  'pentagon_size' : np.random.uniform(0.3,1),
  'rot' : np.random.uniform(0,24),
  'lw' : np.random.uniform(0.1,2.4),
  'dist' : np.random.uniform(0.5,1.6),
  'rot_right': np.random.uniform(0,96),
  'size_right':np.random.uniform(0.5,1.2),
  'rot_both':0,
  'line_randomness':np.random.uniform(1,5)} for i in range(1200)]
if not os.path.exists("data/images/"):
    os.makedirs("data/images/")
image = [simulate_pentagon(**param_list[i]) for i in range(len(param_list))]
pd.DataFrame(param_list).to_csv("data/images/param.txt",sep="\t")
for i in range(len(image)):
    image[i].save("data/images/"+str(i)+".png", format="png")
    
    
# label generation
config = {
    'model_name': "vgg19_bn",
    'batch_size': 32,
    'model_loc': "models/vgg19_bn-1_model.pth"
}
config=munch.Munch(config)
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
net, input_size = define_network(config)
config.input_size = input_size
net.load_state_dict(torch.load(config["model_loc"],
                               map_location=config.device))
transform_img = transforms.Compose([transforms.Resize((config.input_size,config.input_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])
image = torch.stack([transform_img(image[i]) for i in range(len(image))])

out=[]
for i in range(0,1200,32):
    #print(i)
    with torch.no_grad():
        net.eval()
        out.append(net(image[i:(i+32),:,:,:].cuda()).cpu().detach().numpy())
        
if not os.path.exists("data/phenotype/"):
    os.makedirs("data/phenotype/")
    
df = pd.DataFrame(data={'file_name':[f"{i}.png" for i in range(1200)], 'value':np.squeeze(np.concatenate(out))+np.random.normal(0, 0.25, 1200)})
df.iloc[:1000].to_csv('data/phenotype/train.txt',sep='\t',index=False)
df.iloc[1000:1100].to_csv('data/phenotype/valid.txt',sep='\t',index=False)
df.iloc[1100:1200].to_csv('data/phenotype/test.txt',sep='\t',index=False)
