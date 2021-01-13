# Guy: Helpers and references are here: https://www.jeremyjordan.me/semantic-segmentation/
import torch
import torch.nn as nn
import torch.nn.functional as F

class DICE(nn.Module):

    def __init__(self, n_classes, weight):
        super().__init__()
        self.n_classes = n_classes
        print(n_classes)
        if weight is None:
            self.weight = nn.Parameter(torch.tensor([1. / n_classes] * n_classes), requires_grad=False)
        else:
            self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, input, target):
        # Convert one hot
        tar = torch.transpose(F.one_hot(target, num_classes=self.n_classes + 1), 1, 3)[:, 1:, :, :].float()
        # Continue orig:
        smooth = 1.
        #inp = torch.sigmoid(input)
        inp = torch.sigmoid(input[:, 1:, :, :])     # TODO -- on 08.2020
        iflat = inp.flatten(2).contiguous()
        tflat = tar.flatten(2).contiguous()
        intersection = (iflat * tflat).sum(-1)
        loss = (1 - ((2. * intersection + smooth) / (
                (torch.pow(iflat.float(), 2) + torch.pow(tflat.float(), 2)).sum(-1) + smooth)))
        loss = (loss * self.weight).sum(-1)
        loss = loss.mean(-1)  # Guy, ignore weights
        return loss

