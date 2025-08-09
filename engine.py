import torch
import torch.nn as nn
from torch.autograd import Variable
from model import GramMatrix
from preprocessing import postp
from config import style_layers, content_layers
from torch import optim

def synthesizeImage(vgg, style_image, content_image, loss_fn, style_weights, content_weights, max_iter=500, show_iter=100):

    # Image to optimize is a clone of the content image for faster convergence
    opt_img = Variable(content_image.data.clone(), requires_grad=True)
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
            return loss

        optimizer.step(closure)

    return postp(opt_img.data[0].cpu().squeeze())