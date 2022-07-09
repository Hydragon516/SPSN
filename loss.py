import torch
import torch.nn.functional as F

def IoU_loss(preds_list, gt):
    preds = torch.cat(preds_list, dim=1)
    N, C, H, W = preds.shape
    
    min_tensor = torch.where(preds < gt, preds, gt)
    max_tensor = torch.where(preds > gt, preds, gt)
    min_sum = min_tensor.view(N, C, H * W).sum(dim=2)
    max_sum = max_tensor.view(N, C, H * W).sum(dim=2)
    
    loss = 1 - (min_sum / max_sum).mean()
    
    return loss