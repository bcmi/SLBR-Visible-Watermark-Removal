import numpy as np
import torch
from sklearn.metrics import average_precision_score	


class AverageMeter(object):
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


def normPRED(d, eps=1e-2):
    ma = torch.max(d)
    mi = torch.min(d)

    if ma-mi<eps:
        dn = d-mi
    else:
        dn = (d-mi)/(ma-mi)

    return dn

def compute_fPSNR(pred, gt):
    return 

def compute_RMSE(pred, gt, mask, is_w=False):
    if is_w:
        if isinstance(mask, torch.Tensor):
            mse = torch.mean((pred*mask - gt*mask)**2, dim=[1,2,3])
            rmse = mse*np.prod(mask.shape[1:])/(torch.sum(mask, dim=[1,2,3])+1e-6)
            rmse = torch.sqrt(rmse).mean().item()
        elif isinstance(mask, np.ndarray):
            rmse = MSE(pred*mask, gt*mask)*np.prod(mask.shape) / (np.sum(mask)+1e-6)
            rmse = np.sqrt(rmse)
    else:
        if isinstance(mask, torch.Tensor):
            mse = torch.mean((pred - gt)**2, dim=[1,2,3])
            rmse = torch.sqrt(mse).mean().item()

        elif isinstance(mask, np.ndarray):
            rmse = MSE(pred, gt)*np.prod(mask.shape) / (np.sum(mask)+1e-6)
            rmse = np.sqrt(rmse)
    
    return rmse * 256


def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().view(labels.size(0),-1).numpy()
    y_pred = outputs.cpu().detach().view(labels.size(0),-1).numpy()
    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i],y_pred[i]))
    return np.mean(AP)

def compute_IoU(pred, gt, threshold=0.5, eps=1e-5):
    pred = torch.where(pred > threshold, torch.ones_like(pred), torch.zeros_like(pred)).to(pred.device)
    intersection = (pred * gt).sum(dim=[1,2,3])
    union = pred.sum(dim=[1,2,3]) + gt.sum(dim=[1,2,3]) - intersection
    return (intersection / (union+eps)).mean().item()

def MAE(pred, gt):
    if isinstance(pred, torch.Tensor):
        return torch.mean(torch.abs(pred - gt))
    elif isinstance(pred, np.ndarray):
        return np.mean(np.abs(pred-gt))

def FScore(pred, gt, beta2=1.0, threshold=0.5, eps=1e-6, reduce_dims=[1,2,3]):
    if isinstance(pred, torch.Tensor):
        if threshold == -1: threshold = pred.mean().item() * 2
        ones = torch.ones_like(pred).to(pred.device)
        zeros = torch.zeros_like(pred).to(pred.device)
        pred_ = torch.where(pred > threshold, ones, zeros)
        gt = torch.where(gt>threshold, ones, zeros)
        total_num = pred.nelement()

        TP = (pred_ * gt).sum(dim=reduce_dims)
        NumPrecision = pred_.sum(dim=reduce_dims)
        NumRecall = gt.sum(dim=reduce_dims)
        
        precision = TP / (NumPrecision+eps)
        recall = TP / (NumRecall+eps)
        F_beta = (1+beta2)*(precision * recall) / (beta2*precision + recall + eps)
        F_beta = F_beta.mean()
        
    elif isinstance(pred, np.ndarray):
        if threshold == -1: threshold = pred.mean()* 2
        pred_ = np.where(pred > threshold, 1.0, 0.0)
        gt = np.where(gt > threshold, 1.0, 0.0)
        total_num = np.prod(pred_.shape)

        TP = (pred_ * gt).sum()
        NumPrecision = pred_.sum()
        NumRecall = gt.sum()
        
        precision = TP / (NumPrecision+eps)
        recall = TP / (NumRecall+eps)
        F_beta = (1+beta2)*(precision * recall) / (beta2*precision + recall + eps)

    return F_beta

class Fmeasure:
    def __init__(self, n_imgs, beta2=0.3, thresholds=[t/255 for t in range(255,-1, -1)]):
        self.n_imgs = n_imgs
        self.idx = 0
        self.beta2 = beta2
        self.thresholds = thresholds
        self.reset()

    def reset(self):
        if isinstance(self.thresholds, int):
            self.thresholds_fm = np.zeros((self.n_imgs, 1), dtype=np.float)
        elif isinstance(self.thresholds, list):
            self.thresholds_fm = np.zeros((self.n_imgs, len(self.thresholds)), type=np.float)
        self.adp_fm = np.zeros((self.n_imgs,), dtype=np.float) # adaptive threshold
        self.fixed_fm = np.zeros((self.n_imgs,), dtype=np.float) # fixed threshold: 0.5, beta2 = 1.0

    def update(self, pred, gt):
        if isinstance(self.thresholds, int):
            self.thresholds_fm[self.idx] = FScore(pred, gt, beta2=self.beta2, threshold=self.threshold)
        elif isinstance(self.thresholds, list):
            for i, t in enumerate(self.thresholds):
                self.thresholds_fm[self.idx, i] = FScore(pred, gt, beta2=self.beta2, threshold=t)
        # adaptive thresold
        self.adp_fm[self.idx] = FScore(pred, gt, beta2=self.beta2, threshold=-1)
        self.fixed_fm[self.idx] = FScore(pred, gt, beta2=1.0, threshold=0.5)
        self.idx+=1

    def val(self, eps=1e-6):
        column_Fm = self.thresholds.sum(axis=0) / (self.idx+eps)
        mean_Fm = column_Fm.mean()
        max_Fm = column_Fm.max()
        adp_Fm = (self.adp_fm.sum(axis=0) / (self.idx+eps))
        fixed_Fm = (self.fixed_fm.sum(axis=0) / (self.idx+eps))
        return {'meanFm':mean_Fm, 'maxFm':max_Fm, 'adpFm':adp_Fm, 'fixedFm':fixed_Fm}
