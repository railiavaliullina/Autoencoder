from torch import nn
import torch


class MSE_contractive(nn.Module):
    def __init__(self, reg_lambda):
        super().__init__()

        self.reg_lambda = reg_lambda
        self.criterion = nn.MSELoss()

    def forward(self, prediction, target, h=None):
        loss = self.criterion(prediction, target)
        if h is not None:
            input_ = torch.ones(h.size()).cuda()
            h.backward(input_, retain_graph=True)
            der_matrix = target.grad
            reg_loss = self.reg_lambda * torch.sqrt(torch.sum(der_matrix ** 2))
            target.grad.data.zero_()
            loss += reg_loss
        return loss
