def synthesizeImage3(style_image, content_image, loss_fn, 
                     style_weights, content_weights, max_iter=500, 
                     show_iter=100, threshold=1e-4):

    # Image to optimize is a clone of the content image for faster convergence
    opt_img = Variable(content_image.data.clone(), requires_grad=True)
    prev_img = opt_img.data.clone()  # per norma max

    style_layers = ['r11','r21','r31','r41', 'r51']
    content_layers = ['r42']
    loss_layers = style_layers + content_layers

    # Loss functions (MSE, Pearson, etc.)
    loss_fns = [loss_fn()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
    if torch.cuda.is_available():
        loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

    weights = style_weights + content_weights

    # Compute optimization targets
    style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
    content_targets = [A.detach() for A in vgg(content_image, content_layers)]
    targets = style_targets + content_targets

    optimizer = optim.LBFGS([opt_img])  # LBFGS as in Gatys
    n_iter = [0]

    stop_flag = [False]  # segnale di arresto

    while n_iter[0] <= max_iter and not stop_flag[0]:

        def closure():
            optimizer.zero_grad()
            out = vgg(opt_img, loss_layers)
            layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
            loss = torch.sum(torch.stack(layer_losses))
            loss.backward()

            # 🧮 Calcolo delta max (norma infinito degli aggiornamenti)
            delta = (opt_img.data - prev_img).abs()
            delta_norm = torch.max(delta).item()
            if delta_norm < threshold and n_iter[0] > 100:
                for param_group in optimizer.param_groups:
                    print(f"⏹️ Early stopping: Δ = {delta_norm:.3f} < {threshold}, Iteration {n_iter[0]}: Loss = {loss.item()}, Current LR: {np.round(param_group['lr'],3)}")
                stop_flag[0] = True

            prev_img.copy_(opt_img.data)  # aggiornamento per prossimo confronto

            n_iter[0] += 1
            if n_iter[0] % show_iter == 0:
                for param_group in optimizer.param_groups:
                    print(f"Iteration {n_iter[0]}: Current LR: {np.round(param_group['lr'],3)}, Loss = {loss.item()} | Δ = {delta_norm:.6f}")
            return loss

        optimizer.step(closure)

    return postp(opt_img.data[0].cpu().squeeze())
    

prs_out = synthesizeImage3(style_img, content_img, 
					PearsonCorrelationLoss, 
					prs_style_weights, 
					prs_content_weights, 
					max_iter=600, 
                         show_iter=75,
                         threshold = 0.25)



rmse_out = synthesizeImage3(style_img, content_img, 
					GramMSELoss, 
					rmse_style_weights, 
					rmse_content_weights, 
					max_iter=600, 
                         show_iter=75,
                         threshold = 0.25)
        

cos_out = synthesizeImage3(style_img, content_img, 
					CosineSimilarityLoss, 
					cos_style_weights, 
					cos_content_weights,  
					max_iter=600, 
                         show_iter=75,
                         threshold = 0.25)
            
