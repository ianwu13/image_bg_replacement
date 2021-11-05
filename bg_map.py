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

from model import U2NET

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def get_bg_map():

    # Get image location
    image_path = "./images/liz.jpg"

    # Create u2net model object
    weights = "./model/u2net_weights.pth"
    u2net = U2NET(3,1)
    if torch.cuda.is_available():
        u2net.load_state_dict(torch.load(weights))
        u2net.cuda()
    else:
        u2net.load_state_dict(torch.load(weights, map_location='cpu'))
    u2net.eval()

    img = Image.open(image_path)
    orig_dim = img.size
    img = img.resize((320,320))

    transform = transforms.ToTensor()
    inputs_test = transform(img).unsqueeze(0)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    d1,d2,d3,d4,d5,d6,d7= u2net(inputs_test)

    # normalization
    pred = d1[:,0,:,:]
    pred = normPRED(pred)

    # save results to test_results folder
    out = np.squeeze(d1)

    to_im = transforms.ToPILImage()
    im = to_im(out)
    im = im.resize(orig_dim)

    split_img_path = image_path.split('/')
    image_name = split_img_path[len(split_img_path) - 1]
    map_name = image_name.split('.')[0] + "-map." + image_name.split('.')[1]
    im.save(map_name)

if __name__ == "__main__":
    get_bg_map()