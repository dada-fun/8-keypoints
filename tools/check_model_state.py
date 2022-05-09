import torch


model = torch.load('../ckpt/model_best.pth', map_location='cpu')
model2 = torch.load('../ckpt/epoch_110_aic_coco.pth', map_location='cpu')
print(model)

model