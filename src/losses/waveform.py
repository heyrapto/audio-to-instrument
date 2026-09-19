import torch.nn.functional as F

def waveform_loss(predicted, target):
    return F.l1_loss(predicted, target)
