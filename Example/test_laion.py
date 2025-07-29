import json
import torch
import torch.nn as nn
from PIL import Image
import os
from os.path import expanduser  # pylint: disable=import-outside-toplevel
import open_clip
from pathlib import Path
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

# !pip install open-clip-torch is needed

####################################
# LAION Instantiation
####################################

def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m

amodel= get_aesthetic_model(clip_model="vit_l_14")
amodel.eval()

model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')

############################################
# Load Images and Compute Scores
############################################

img_dir = Path("output")
scores = {}

for path in sorted(img_dir.glob("*.jpg")):
    base, metric = path.stem.rsplit("_", 1)
    metric = {"cos": "cosine", "prs": "pearson", "rmse": "rmse"}.get(metric, metric)
    img = Image.open(path)
    image = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        f = model.encode_image(image)
        f /= f.norm(dim=-1, keepdim=True)
        pred = amodel(f).item()
    scores.setdefault(base, {"rmse": None, "cosine": None, "pearson": None})[metric] = pred

with open("laion_scores.json", "w") as f:
    json.dump(scores, f, indent=2)