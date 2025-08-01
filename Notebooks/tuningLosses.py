import json
import os
import torch
from torch.autograd import Variable
from torch import optim
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torch
import random
import numpy as np
from statistics import mean

# Imposta il seed per tutti i moduli
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Per CUDA
torch.cuda.manual_seed(seed)

# Migliora il determinismo del comportamento
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

img_size = 512
prep = transforms.Compose([transforms.Resize(img_size),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])

content_path = "pictures_new_weights/content"
style_path = "pictures_new_weights/style"

content_imgs = [Image.open(os.path.join(content_path, img)) for img in os.listdir(content_path)]
style_imgs = [Image.open(os.path.join(style_path, img)) for img in os.listdir(style_path)]
content_imgs = [Variable(prep(img).unsqueeze(0).cuda()) for img in content_imgs]
style_imgs = [Variable(prep(img).unsqueeze(0).cuda()) for img in style_imgs]


class VGG(nn.Module):
    def __init__(self, pool='avg'):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])

        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])

        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])

        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])

        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        
        return [out[key] for key in out_keys]

class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2))
        G.div_(h*w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)
    
class PearsonCorrelationLoss(nn.Module):
    def forward(self, x, y):
        """
        Pearson correlation loss.
        """
        
        x_gram = GramMatrix()(x)

        x_flat = x_gram.view(-1).float()
        y_flat = y.view(-1).float()

        mean_x = torch.mean(x_flat)
        mean_y = torch.mean(y_flat)

        xm = x_flat - mean_x
        ym = y_flat - mean_y

        numer = torch.dot(xm, ym)
        denom = torch.sqrt(torch.dot(xm, xm) * torch.dot(ym, ym))
        eps = 1e-8
        r = numer / (denom + eps)
        distance = 1.0 - r
        return (distance)

class CosineSimilarityLoss(nn.Module):
    def forward(self, x, y):
        """
        Cosine similarity loss.
        """
        x_gram = GramMatrix()(x)

        x_flat = x_gram.view(-1).float()
        y_flat = y.view(-1).float()
        # Dot‐product
        dot_xy = torch.dot(x_flat, y_flat)
        # Norme L2
        norm_x = torch.norm(x_flat, p=2)
        norm_y = torch.norm(y_flat, p=2)
        eps = 1e-8
        cos_sim = dot_xy / (norm_x * norm_y + eps)
        return (1.0 - cos_sim)


postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])
postpb = transforms.Compose([transforms.ToPILImage()])

def postp(tensor): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1
    t[t<0] = 0
    img = postpb(t)
    return img

vgg = VGG()
vgg.load_state_dict(torch.load('/home/dirita/projectwork/GramStyleAnalysis/style_evaluation/vgg_conv.pth'))
#vgg.load_state_dict(torch.load('vgg_conv.pth'))

for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()

def synthesizeImage(style_image, content_image, loss_fn, style_weights, content_weights, max_iter=500, show_iter=100):

    # Image to optimize is a clone of the content image for faster convergence
    opt_img = Variable(content_image.data.clone(), requires_grad=True)
    style_layers = ['r11','r21','r31','r41', 'r51']
    content_layers = ['r42']
    loss_layers = style_layers + content_layers

    # Here the used loss functions are instantiated (MSE, Pearson, Cosine)
    loss_fns = [loss_fn()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
    if torch.cuda.is_available():
        loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

    weights = style_weights + content_weights

    #compute optimization targets
    style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
    content_targets = [A.detach() for A in vgg(content_image, content_layers)]
    targets = style_targets + content_targets

    #run style transfer
    optimizer = optim.LBFGS([opt_img]) # LBFGS as in the original gatys implementation
    n_iter=[0]

    while n_iter[0] <= max_iter:

        def closure():
            optimizer.zero_grad()
            out = vgg(opt_img, loss_layers)
            layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
            loss = torch.sum(torch.stack(layer_losses))
            loss.backward()
            n_iter[0]+=1
            if n_iter[0] % show_iter == 0:
                print(f"Iteration {n_iter[0]}: Loss = {loss.item()}")
                print("Layer losses:", [l.item() for l in layer_losses])
            return loss

        optimizer.step(closure)

    return postp(opt_img.data[0].cpu().squeeze())

# As in Gatys et al. 2016
rmse_style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
rmse_content_weights = [1e0]

def compute_coefficients(content_img, style_img, vgg, loss_fn):
    opt_img = Variable(content_img.data.clone(), requires_grad=True)
    style_layers = ['r11','r21','r31','r41', 'r51']
    content_layers = ['r42']
    loss_layers = style_layers

    loss_fns = [loss_fn()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
    loss_fns_rmse = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
    
    if torch.cuda.is_available():
        loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
        loss_fns_rmse = [loss_fn.cuda() for loss_fn in loss_fns_rmse]

    targets = [GramMatrix()(A).detach() for A in vgg(style_img, style_layers)]

    # Compute the output of the VGG model
    out = vgg(opt_img, loss_layers)

    coeff_1 = [loss_fns[a](A, targets[a]).item() for a,A in enumerate(out)]
    coeff_rmse = [rmse_style_weights[a] * loss_fns_rmse[a](A, targets[a]).item() for a,A in enumerate(out)]

    new_coeff = [rmse/one if one != 0 else 0 for rmse, one in zip(coeff_rmse, coeff_1)]
    print("new coefficients:", new_coeff)
    return new_coeff
    

output_path = "synthetized_images_dynamic_weights"

coeff_pcc = json.load(open("newLossesWeights.json"))["pearson_coeff"]
coeff_cos = json.load(open("newLossesWeights.json"))["cos_coeff"]

i = 0
tot = len(content_imgs) * len(style_imgs)

for idx_c, content_img in enumerate(content_imgs):
    for idx_s, style_img in enumerate(style_imgs):

        prs_out = synthesizeImage(style_img, content_img, PearsonCorrelationLoss, coeff_pcc, [1e0], max_iter=600, show_iter=200)
        prs_out.save(f"{output_path}/style{idx_s}_content{idx_c}_prs.jpg")

        cos_out = synthesizeImage(style_img, content_img, CosineSimilarityLoss, coeff_cos, [1e0], max_iter=600, show_iter=200)
        cos_out.save(f"{output_path}/style{idx_s}_content{idx_c}_cos.jpg")
        
        print(f"Processed {i+1}/{tot} images")
        i += 1
