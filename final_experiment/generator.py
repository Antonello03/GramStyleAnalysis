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
from config import compute_ratio_style_weights, style_layers, content_layers
from preprocessing import prep
from engine import synthesizeImage, synthesizeImage3

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
vgg_weights_path = '/home/dirita/projectwork/GramStyleAnalysis/style_evaluation/vgg_conv.pth'
vgg = VGG()
vgg.load_state_dict(torch.load(vgg_weights_path))

for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()

# Content and Style Images, Output Directory
content_dir = 'final_experiment/content/'
style_dir = 'final_experiment/style/'
output_dir = 'final_experiment/output_alt_newlr'


content_imgs = [Image.open(os.path.join(content_dir, f)) for f in os.listdir(content_dir) if f.endswith('.jpg')]
content_imgs = [Variable(prep(img).unsqueeze(0).cuda()) for img in content_imgs]
style_imgs = [Image.open(os.path.join(style_dir, f)) for f in os.listdir(style_dir) if f.endswith('.jpg')]
style_imgs = [Variable(prep(img).unsqueeze(0).cuda()) for img in style_imgs]

content_names = [os.path.splitext(f)[0] for f in os.listdir(content_dir) if f.endswith('.jpg')]
style_names = [os.path.splitext(f)[0] for f in os.listdir(style_dir) if f.endswith('.jpg')]

i = 0
threshold = 0.5
style_loss_weight = 6.5
TOTAL_SAMPLES = len(content_imgs) * len(style_imgs)
start = time.perf_counter()

def fmt_time(sec):
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

losses_list = [
   ("rmse", GramMSELoss, 1.5),
   ("prs",  PearsonCorrelationLoss, 1.5),
   ("cos",  CosineSimilarityLoss, 1.75)
]

methods = [suffix for suffix, _, _ in losses_list]

method_times = {method: 0.0 for method in methods}

for idx_c, content_img in enumerate(content_imgs[:2]):
    for idx_s, style_img in enumerate(style_imgs[:2]):
        i += 1
        c_name, s_name = content_names[idx_c], style_names[idx_s]

        for suffix, loss_cls, lr in losses_list:
            start_loss = time.perf_counter()

            sw, cw = compute_ratio_style_weights(vgg, content_img, style_img, style_layers, content_layers, loss_cls, style_loss_weight=style_loss_weight, verbose=False, individual_layer_weights_by_loss= True) # different method
            synthesizeImage3(
                vgg, style_img, content_img, loss_cls, sw, cw,
                max_iter=500, show_iter=1000, threshold=threshold, lr=lr
            ).save(f"{output_dir}/{s_name}_{c_name}_{suffix}.jpg")

            method_times[suffix] += time.perf_counter() - start_loss

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

avg_times = {
    method: method_times[method] / i
    for method in methods
}

print("\nAverage times per method:")
for method in methods:
    print(f"  {method}: {avg_times[method]:.3f} seconds")