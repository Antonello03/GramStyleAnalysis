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

folder = "laion_scores"
file_names = ["laion_scores_style_loss_0_5.json", "laion_scores_style_loss_1.json", "laion_scores_style_loss_1_5.json", "laion_scores_style_loss_2.json"]
folder_paths = ["output_alt/0.5", "output_alt/1", "output_alt/1.5", "output_alt/2"]

for file_name, folder_path in zip(file_names, folder_paths):

    img_dir = Path(folder_path)
    scores = {}

    for path in sorted(img_dir.glob("*.jpg")):
        base, metric = path.stem.rsplit("_", 1)
        metric = {"cos": "cosine", "prs": "pearson", "rmse": "rmse"}.get(metric, metric)
        img = Image.open(path)
        score = aesthetic_score(img)
        scores.setdefault(base, {"cosine": None, "pearson": None, "rmse": None})[metric] = score.item()

    with open(folder + "/" + file_name, "w") as f:
        json.dump(scores, f, indent=2)
