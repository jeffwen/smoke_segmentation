from torch import nn


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2. * (pred * truth).double().sum() + 1) / (pred.double().sum() + truth.double().sum() + 1)

        return bce_loss + (1 - dice_coef)


# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
def jaccard_index(input, target):

    intersection = (input*target).long().sum().data.cpu()[0]
    union = input.long().sum().data.cpu()[0] + target.long().sum().data.cpu()[0] - intersection

    if union == 0:
        return float('nan')
    else:
        return float(intersection) / float(max(union, 1))
    
# https://discuss.pytorch.org/t/understanding-different-metrics-implementations-iou/85817
def get_iou(outputs, labels, EPS=1e-6):
    outputs = outputs.int()
    labels = labels.int()
    
    intersection = (outputs & labels).float().sum((1, 2)) 
    union = (outputs | labels).float().sum((1, 2))

    iou = (intersection + EPS) / (union + EPS)

    return iou.mean()


# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(input, target):
    num_in_target = input.size(0)

    smooth = 1.

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    loss = (2. * intersection + smooth) /(pred.sum(1) + truth.sum(1) + smooth)
    
    return loss.mean().item()
