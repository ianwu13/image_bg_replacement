import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import cv2
from model import U2NET


def get_bg_map(img):
    # Create u2net model object and load weights
    weights = "./model/u2net_weights.pth"
    u2net = U2NET(3,1)
    if torch.cuda.is_available():
        u2net.load_state_dict(torch.load(weights))
        u2net.cuda()
    else:
        u2net.load_state_dict(torch.load(weights, map_location='cpu'))
    u2net.eval()

    # Save original dimensions
    orig_dim = img.shape

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
    out = np.squeeze(d1[:,0,:,:]).detach().numpy()

    #resize to original dimensions
    out = cv2.resize(out, (orig_dim[1], orig_dim[0]))

    return out
