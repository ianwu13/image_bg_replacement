import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from u2net import U2NET # full size version 173.6 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def main():

    # --------- 1. get image and model path ---------
    image_path = "liz.jpg"
    model_dir = "u2net_weights.pth"

    # --------- 3. model define ---------
    print("...load U2NET---173.6 MB")
    net = U2NET(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    img = Image.open(image_path)
    orig_dim = img.size
    img = img.resize((320,320))

    transform = transforms.ToTensor()
    inputs_test = transform(img).unsqueeze(0)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

    # normalization
    pred = d1[:,0,:,:]
    pred = normPRED(pred)

    # save results to test_results folder
    out = np.squeeze(d1)

    to_im = transforms.ToPILImage()
    im = to_im(out)
    im = im.resize(orig_dim)

    map_name = image_path.split('.')[0] + "-map." + image_path.split('.')[1]
    im.save(map_name)

if __name__ == "__main__":
    main()