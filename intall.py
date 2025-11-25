import torch, torchvision
m = torchvision.models.resnet18(weights="DEFAULT").eval().cuda()
dummy = torch.randn(1, 3, 224, 224, device="cuda")
with torch.no_grad():
    out = m(dummy)
print("Inference OK on:", next(m.parameters()).device, "output shape:", out.shape)
