import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import glob
import cv2
from model import U2NET

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def get_bg_map(img):
    # Create u2net model object
    weights = "./model/u2net_weights.pth"
    u2net = U2NET(3,1)
    if torch.cuda.is_available():
        u2net.load_state_dict(torch.load(weights))
        u2net.cuda()
    else:
        u2net.load_state_dict(torch.load(weights, map_location='cpu'))
    u2net.eval()

    # Save original dimensions
    orig_dim = img.shape[0:2]

    # Get input tensor 
    img = cv2.resize(img, (320,320))
    transform = transforms.ToTensor()
    inputs = transform(img).unsqueeze(0)
    if torch.cuda.is_available():
        inputs = Variable(inputs.cuda())
    else:
        inputs = Variable(inputs)

    # Run model to generate saliency map
    d1,d2,d3,d4,d5,d6,d7= u2net(inputs[:,0:3,:,:])

    # TODO

    # normalization
    pred = d1[:,0,:,:]
    pred = normPRED(pred)

    # save results to test_results folder
    out = d1.detach().numpy() # np.squeeze(d1)
    print(out.shape)

    #to_im = transforms.ToPILImage()
    #im = to_im(out)
    out = cv2.resize(out, orig_dim)

    split_img_path = image_path.split('/')
    image_name = split_img_path[len(split_img_path) - 1]
    map_name = image_name.split('.')[0] + "-map." + image_name.split('.')[1]
    cv2.imwrite(map_name, out)

if __name__ == "__main__":
    # Get image location
    image_path = "./images/liz.jpg"
    img = cv2.imread(image_path)
    get_bg_map(img)