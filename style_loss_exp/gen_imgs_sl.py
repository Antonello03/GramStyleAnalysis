# Imports
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import os
import torch
import time, datetime
from torch.autograd import Variable
from PIL import Image
import random
import numpy as np
from model import VGG
from losses import GramMSELoss, PearsonCorrelationLoss, CosineSimilarityLoss
from config import compute_ratio_style_weights
from config import style_layers, content_layers, max_iter
from preprocessing import prep, postp
from engine import synthesizeImage

# Parameters for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Model Definition
vgg_weights_path = '/home/antonello03/workspace/workspace/AI/ProjectWork/gramstyle-thesis/Notebooks/vgg_conv.pth'
vgg = VGG()
vgg.load_state_dict(torch.load(vgg_weights_path))

for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()

# Content and Style Images, Output Directory
content_dir = 'style_loss_exp/content/'
style_dir = 'style_loss_exp/style/'
output_dir = 'style_loss_exp/output'


content_imgs = [Image.open(os.path.join(content_dir, f)) for f in os.listdir(content_dir) if f.endswith('.jpg')]
content_imgs = [Variable(prep(img).unsqueeze(0).cuda()) for img in content_imgs]
style_imgs = [Image.open(os.path.join(style_dir, f)) for f in os.listdir(style_dir) if f.endswith('.jpg')]
style_imgs = [Variable(prep(img).unsqueeze(0).cuda()) for img in style_imgs]

content_names = [os.path.splitext(f)[0] for f in os.listdir(content_dir) if f.endswith('.jpg')]
style_names = [os.path.splitext(f)[0] for f in os.listdir(style_dir) if f.endswith('.jpg')]



i = 0
TOTAL_SAMPLES = len(content_imgs) * len(style_imgs) * 4
start = time.perf_counter()

def fmt_time(sec):
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

losses_list = [
    ("rmse", GramMSELoss),
    ("prs",  PearsonCorrelationLoss),
    ("cos",  CosineSimilarityLoss)
]

for style_loss_weight in [0.5, 1, 1.5, 2]:
    for idx_c, content_img in enumerate(content_imgs):
        for idx_s, style_img in enumerate(style_imgs):
            i += 1
            c_name, s_name = content_names[idx_c], style_names[idx_s]

            for suffix, loss_cls in losses_list:
                sw, cw = compute_ratio_style_weights(vgg, content_img, style_img, style_layers, content_layers, loss_cls, style_loss_weight=style_loss_weight, verbose=False)
                synthesizeImage(
                    vgg, style_img, content_img, loss_cls, sw, cw,
                    max_iter=max_iter, show_iter=max_iter//3
                ).save(f"{output_dir}/{style_loss_weight}/{s_name}_{c_name}_{suffix}.jpg")

            elapsed = time.perf_counter() - start
            ips = i / elapsed if elapsed > 0 else 0.0
            remaining = (TOTAL_SAMPLES - i) / ips if ips > 0 else float("inf")
            pct = min(100.0, i * 100.0 / TOTAL_SAMPLES)
            eta_clock = (datetime.datetime.now() + datetime.timedelta(seconds=remaining)).strftime("%Y-%m-%d %H:%M:%S") if ips > 0 else "calculating..."

            print(
                f"[{i}/{TOTAL_SAMPLES}  {pct:5.1f}%] "
                f"elapsed {fmt_time(elapsed)} | remaining {fmt_time(remaining)} | ETA {eta_clock}",
                end="\r", flush=True
            )
