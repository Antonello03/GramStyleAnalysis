import os
import torch
from tqdm import tqdm
from torch.autograd import Variable
from torch import optim
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset
import nest_asyncio; nest_asyncio.apply()

model_dir  = "/home/antonello03/gdrive/Colab Notebooks/UNI/ProjectWork/"
image_path = "/mnt/e/synthesized-wikiart"

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

# pre and post processing for images
img_size = 512
prep = transforms.Compose([transforms.Resize(img_size),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])
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

# Model Preparation

vgg = VGG()
vgg.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))

for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()

# Dataset Preparation

stream_ds = load_dataset(
    "huggan/wikiart",
    split="train",        # the dataset has a single split called “train”
    streaming=True        # key flag – nothing is cached locally until you touch it
)

num_threads = num_threads = min(32, (os.cpu_count() or 1) + 4)
stream_ds = stream_ds.decode(num_threads=num_threads)
it = iter(stream_ds)

def synthesizeImage(style_image, content_image, loss_fn, style_weights, content_weights, max_iter=500, show_iter=100):
    opt_img = Variable(content_image.data.clone(), requires_grad=True)
    style_layers = ['r11','r21','r31','r41', 'r51']
    content_layers = ['r42']
    loss_layers = style_layers + content_layers
    loss_fns = [loss_fn()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
    if torch.cuda.is_available():
        loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

    weights = style_weights + content_weights

    #compute optimization targets
    style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
    content_targets = [A.detach() for A in vgg(content_image, content_layers)]
    targets = style_targets + content_targets

    #run style transfer
    optimizer = optim.LBFGS([opt_img])
    n_iter=[0]

    while n_iter[0] <= max_iter:

        def closure():
            optimizer.zero_grad()
            out = vgg(opt_img, loss_layers)
            layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
            loss = torch.sum(torch.stack(layer_losses))
            loss.backward()
            n_iter[0]+=1
            if n_iter[0]%show_iter == (show_iter-1):
                print(f"Iteration {n_iter[0]+1}, loss: {loss.item()}")
                #print([f"{loss_layers[li]}: {l.item()}" for li, l in enumerate(layer_losses)])
            return loss

        optimizer.step(closure)

    return postp(opt_img.data[0].cpu().squeeze())

content_img = Image.open("Tuebingens.jpg")
content_img = Variable(prep(content_img).unsqueeze(0).cuda())

rmse_style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
rmse_content_weights = [1e0]
prs_style_weights = [5e7/n**2 for n in [64,128,256,512,512]]
prs_content_weights = [1e-3]
cos_style_weights = [5e7/n**2 for n in [64,128,256,512,512]]
cos_content_weights = [1e-3]

i = 0
max_iter = 500
type_res = f"512_res"


for sample in tqdm(it):
    i += 1
    img = sample["image"]
    img.save(f"{image_path}/{i}_style_{type_res}.jpg")
    img = prep(img)
    img = Variable(img.unsqueeze(0).cuda())
    rmse_out = synthesizeImage(img, content_img, GramMSELoss, rmse_style_weights, rmse_content_weights, max_iter=max_iter, show_iter=max_iter+1)
    rmse_out.save(f"{image_path}/{i}_rmse_{type_res}.jpg")
    prs_out = synthesizeImage(img, content_img, PearsonCorrelationLoss, prs_style_weights, prs_content_weights, max_iter=max_iter, show_iter=max_iter+1)
    prs_out.save(f"{image_path}/{i}_prs_{type_res}.jpg")
    cos_out = synthesizeImage(img, content_img, CosineSimilarityLoss, cos_style_weights, cos_content_weights, max_iter=max_iter, show_iter=max_iter+1)
    cos_out.save(f"{image_path}/{i}_cos_{type_res}.jpg")