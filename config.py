from model import GramMatrix
import torch
from losses import GramMSELoss

style_layers = ['r11','r21','r31','r41', 'r51']
content_layers = ['r42']
max_iter = 500 
img_size = 512


# As in Gatys et al. 2016
rmse_style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
rmse_content_weights = [1e0]

# Defined manually to match content/style ratio of rmse pictures
prs_style_weights = [5e7/n**2 for n in [64,128,256,512,512]]
prs_content_weights = [1e-3]
cos_style_weights = [5e7/n**2 for n in [64,128,256,512,512]]
cos_content_weights = [1e-3]


def compute_ratio_style_weights(vgg, 
                                content_img, 
                                style_img, 
                                style_layers, 
                                content_layers, 
                                style_loss_fn,
                                style_loss_weight = 1/3,  # Peso relativo della style loss rispetto alla content loss
                                verbose = False):  
    """
    Calcola i pesi relativi per ciascun livello di stile in una rete di style transfer.

    L'obiettivo è bilanciare la style loss rispetto alla content loss, assegnando a ogni livello
    di stile un peso proporzionale alla sua importanza e alla sua risposta. Questo permette di
    ottenere uno stile più coerente e controllato durante l'ottimizzazione.

    Args:
        vgg (callable): Funzione o modello che estrae le feature maps da un'immagine.
        content_img (Tensor): Immagine di contenuto (batch_size x 3 x H x W).
        style_img (Tensor): Immagine di stile (batch_size x 3 x H x W).
        style_layers (list): Indici o nomi dei layer da cui estrarre le feature di stile.
        content_layers (list): Indici o nomi dei layer da cui estrarre le feature di contenuto.
        style_loss_fn (callable): Funzione di loss da usare per confrontare le feature di stile.
                                  Può essere GramMSELoss o una funzione personalizzata.
        style_loss_weight (float, optional): Fattore di bilanciamento tra style e content loss.
                                             Default = 1/3.
        verbose (bool, optional): Se True, stampa dettagli intermedi per debugging.

    Returns:
        scaled_style_weights (list of float): Pesi normalizzati per ciascun livello di stile.
        rmse_content_weights (list of float): Pesi per la content loss (attualmente fissi).
    """
  
    import torch.nn as nn

    rmse_content_weights = [1.0]
    eps = 1e-9
    layer_channels = [64, 128, 256, 512, 512]

    # Estrai feature maps
    content_feats = vgg(content_img, content_layers)
    style_feats = vgg(style_img, style_layers)

    # Content loss con feature fittizie costanti
    content_loss = 0.0
    for feat in content_feats:
        mean_val = feat.mean()
        fake_feat = torch.full_like(feat, mean_val)
        content_loss += nn.MSELoss()(feat, fake_feat)
    if verbose:
        print(f"loss contenuto mse: {content_loss.item():.3e}")    

    # Style loss per livello
    style_losses = []

    for feat in style_feats:
        gram = GramMatrix()(feat)
        # Calcola la deviazione standard per ogni canale
        channel_std = feat.std(dim=(2, 3), keepdim=True)
        ##b, c, h, w = feat.size()
        ##channel_means = feat.mean(dim=(2, 3), keepdim=True)
        ###std = feat.mean() * 0.05
        ##noise = torch.randn_like(feat) * std
        ##fake_feat = channel_means.expand(b, c, h, w).clone() + noise
        # Genera rumore con media 0 e deviazione standard calcolata
        noise = torch.randn_like(feat) * channel_std
        
        # Fake feature con media 0 e deviazione standard come quella di feat
        fake_feat = noise


        fake_gram = GramMatrix()(fake_feat)

        if isinstance(style_loss_fn(), GramMSELoss):
            loss = nn.MSELoss()(gram, fake_gram)
        else:
            loss = style_loss_fn()(feat, fake_gram)

        style_losses.append(loss.item())

    if verbose:
        print(f"loss stile per livello: {[f'{l:.3e}' for l in style_losses]}")

    # Calcolo dei pesi finali per ciascun livello
    scaled_style_weights = []
    tot_style_loss = sum(style_losses)
    squared_channels = [n**2 for n in layer_channels]
    # Step 2: Sum of all squared channels
    total_ch = sum(squared_channels)

    for i in range(len(style_losses)):
        #peso bsato sulla dimensione della feature map rappostato alla loss totale
        style_loss_i = tot_style_loss*(squared_channels[i])/total_ch
        weight_i = (style_loss_weight * content_loss.item()) / (style_loss_i + eps)
        #scaled_weight_i = weight_i * (1 / (n_i**2 + eps))
        scaled_weight_i = weight_i
        scaled_style_weights.append(scaled_weight_i)

    if verbose:
        print(f"scaled_style_weights: {[f'{w:.3e}' for w in scaled_style_weights]}")
        #print(f"rmse_content_weights: {rmse_content_weights}")

    return scaled_style_weights, rmse_content_weights
