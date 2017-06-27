import torch
checkpoint = torch.load('europarl_cpu.pt')
model = checkpoint['model']
for pname in model:
    print(pname)
    assert isinstance(model[pname], torch.FloatTensor)
