import torch
import torch.nn as nn


# SR : Segmentation Result
# GT : Ground Truth
def confusion(SR, GT):
    SR = SR.view(-1)
    GT = GT.view(-1)
    confusion_vector = SR / GT

    TP = torch.sum(confusion_vector == 1).item()
    FP = torch.sum(confusion_vector == float('inf')).item()
    TN = torch.sum(torch.isnan(confusion_vector)).item()
    FN = torch.sum(confusion_vector == 0).item()

    return TP, FP, TN, FN


def get_result(SR, GT, threshold=0.5):  # gpu版本
    SR[SR > threshold] = 1
    SR[SR < 1] = 0
    TP, FP, TN, FN = confusion(SR, GT)

    acc = (TP + TN) / (float(TP + TN + FP + FN) + 1e-6)
    sensitivity = TP / (float(TP + FN) + 1e-6)
    specificity = TN / (float(TN + FP) + 1e-6)
    precision = TP / (float(TP + FP) + 1e-6)
    F1 = 2 * sensitivity * precision / (sensitivity + precision + 1e-6)
    JS = TP / (float(FP + TP + FN) + 1e-6)
    DC = 2 * TP / (float(FP + 2 * TP + FN) + 1e-6)
    IOU = TP / (float(TP + FP + FN) + 1e-6)

    return acc, sensitivity, specificity, precision, F1, JS, DC, IOU


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):
        num = targets.size(0)
        smooth = 1

        # probs = F.sigmoid(logits)
        # m1 = probs.view(num, -1)
        # m2 = targets.view(num, -1)
        m1 = probs
        m2 = targets
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score
