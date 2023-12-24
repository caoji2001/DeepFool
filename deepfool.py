import torch


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):
    with torch.no_grad():
        pred = net(image).flatten().numpy()
    I = pred.argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]

    w = torch.zeros_like(image, dtype=torch.float64)
    r_tot = torch.zeros_like(image, dtype=torch.float64)

    pert_image = image.detach().clone()
    x = pert_image.clone().detach().requires_grad_(True)
    pert_pred = net(x)

    k_i = label
    loop_i = 0
    while k_i == label and loop_i < max_iter:
        pert = float('inf')
        pert_pred[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.detach().clone()

        for k in range(1, num_classes):
            pert_pred[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.detach().clone()

            w_k = cur_grad - grad_orig
            f_k = (pert_pred[0, I[k]] - pert_pred[0, I[0]]).detach().clone()

            pert_k = torch.abs(f_k) / torch.norm(w_k.flatten(), p=2)

            if pert_k < pert:
                pert = pert_k
                w = w_k

        r_i =  (pert+1e-4) * w / torch.norm(w.flatten(), p=2)
        r_tot = (r_tot + r_i).type(torch.float32)

        pert_image = image + (1+overshoot) * r_tot

        x = pert_image.clone().detach().requires_grad_(True)
        pert_pred = net(x)
        k_i = pert_pred.detach().flatten().argmax().item()
        loop_i += 1

    r_tot = (1+overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image
