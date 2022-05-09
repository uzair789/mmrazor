# Author Uzair.
import torch
import torch.nn as nn
# import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class NormalizedLogitMap(nn.Module):
    """mmrazor version of nlm loss for disstilling one stage detectors.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
        normalization (Boolean): Whether or not to run l2 normalization.
    """

    def __init__(
        self,
        tau=1.0,
        loss_weight=1.0,
        normalization=True
    ):
        super(NormalizedLogitMap, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.normalization = normalization

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        N, C, H, W = preds_S.shape

        class_output_teacher = preds_T
        class_output = preds_S
        batch_size = class_output.shape[0]

        c_loss_distill = 0
        for i in range(batch_size):
            if self.normalization:
                class_teacher = class_output_teacher[i] / torch.norm(class_output_teacher[i])
                class_student = class_output[i] / torch.norm(class_output[i])
            else:
                class_teacher = class_output_teacher[i]
                class_student = class_output[i]
            c_loss = torch.norm(class_teacher - class_student)
            c_loss_distill += c_loss

        loss = self.loss_weight * (c_loss_distill/batch_size)

        '''
        softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / self.tau, dim=1)

        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(preds_T.view(-1, W * H) / self.tau) -
                         softmax_pred_T *
                         logsoftmax(preds_S.view(-1, W * H) / self.tau)) * (
                             self.tau**2)

        loss = self.loss_weight * loss / (C * N)
        '''
        return loss
