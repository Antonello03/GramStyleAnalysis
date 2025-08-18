import torch
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

def get_aesthetic_scores(images):
    """
    Instantiate LAION model then preprocess the images to obtain aesthetic score
    """

    model, preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(torch.bfloat16).cuda()

    scores = []

    for image in images:
        pixel_values = (
            preprocessor(images=image, return_tensors="pt")
            .pixel_values.to(torch.bfloat16)
            .cuda()
        )
        with torch.inference_mode():
            score = model(pixel_values).logits.squeeze().float().cpu().numpy()
        scores.append(score)
    return scores

