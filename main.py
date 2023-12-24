import os
from PIL import Image
import numpy as np
import timm
import torch

from deepfool import deepfool


if __name__ == '__main__':
    os.makedirs('input', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    model = timm.create_model('resnet50.a1_in1k', pretrained=True)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    transforms_mean = transforms.transforms[-1].mean.reshape(3,1,1)
    transforms_std = transforms.transforms[-1].std.reshape(3,1,1)

    for img_name in ['0_label91.jpg', '1_label171.jpg', '2_label980.jpg', '3_label218.jpg', '4_label19.jpg']:
        img = Image.open(os.path.join('imagenet-1k-val', img_name))
        img = transforms(img).unsqueeze(0)

        pred = model(img)
        top5_probabilities, top5_class_indices = torch.topk(pred.softmax(dim=1) * 100, k=5)
        print('=' * 20)
        print(f'img name: {img_name}')
        print(f'raw img top5_probabilities: {top5_probabilities.detach().tolist()[0]}')
        print(f'raw img top5_class_indices: {top5_class_indices.tolist()[0]}')

        r_tot, loop_i, label, k_i, pert_image = deepfool(img, model)
        top5_probabilities, top5_class_indices = torch.topk(model(pert_image).softmax(dim=1) * 100, k=5)

        print(f'loop_i: {loop_i}')
        print(f'pert img top5_probabilities: {top5_probabilities.detach().tolist()[0]}')
        print(f'pert img top5_class_indices: {top5_class_indices.tolist()[0]}')
        print('=' * 20)

        input = ((img.squeeze(0) * transforms_std + transforms_mean) * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        input = Image.fromarray(input)
        input.save(os.path.join('input', img_name))

        output = ((pert_image.squeeze(0) * transforms_std + transforms_mean) * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        output = Image.fromarray(output)
        output.save(os.path.join('output', img_name))
