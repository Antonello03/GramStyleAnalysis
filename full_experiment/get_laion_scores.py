import json
import torch
from PIL import Image
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from pathlib import Path


####################################
# LAION 2.5 Instantiation
####################################

model, preprocessor = convert_v2_5_from_siglip(
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model = model.to(torch.bfloat16).cuda()

def aesthetic_score(image):
    """PIL Image -> aesthetic score"""
    pixel_values = (
        preprocessor(images=image, return_tensors="pt")
        .pixel_values.to(torch.bfloat16)
        .cuda()
    )
    pixel_values
    with torch.inference_mode():
        score = model(pixel_values).logits.squeeze().float().cpu().numpy()
    return score

############################################
# Load Images and Compute Scores
############################################

file_name = "laion_scores_full_experiment_preliminary_results.json"
folder_path = "output"

img_dir = Path(folder_path)
scores = {}

for path in sorted(img_dir.glob("*.jpg")):
    base, metric = path.stem.rsplit("_", 1)
    metric = {"cos": "cosine", "prs": "pearson", "rmse": "rmse"}.get(metric, metric)
    img = Image.open(path)
    score = aesthetic_score(img)
    scores.setdefault(base, {"cosine": None, "pearson": None, "rmse": None})[metric] = score.item()

with open(file_name, "w") as f:
    json.dump(scores, f, indent=2)
