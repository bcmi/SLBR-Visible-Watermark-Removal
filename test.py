from __future__ import print_function, absolute_import

import argparse
import torch
import os
from math import log10
import cv2
import numpy as np

torch.backends.cudnn.benchmark = True

import datasets as datasets
import src.models as models
from options import Options
import torch.nn.functional as F
import pytorch_ssim
from evaluation import compute_IoU, FScore, AverageMeter, compute_RMSE, normPRED
from skimage.measure import compare_ssim as ssim
import time


def is_dic(x):
    return type(x) == type([])



def tensor2np(x, isMask=False):
    if isMask:
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        x = ((x.cpu().detach()))*255
    else:
        x = x.cpu().detach()
        mean = 0
        std = 1
        x = (x * std + mean)*255
		
    return x.numpy().transpose(0,2,3,1).astype(np.uint8)

def save_output(inputs, preds, save_dir, img_fn, extra_infos=None,  verbose=False, alpha=0.5):
    outs = []
    image, bg_gt,mask_gt = inputs['I'], inputs['bg'], inputs['mask']
    image = cv2.cvtColor(tensor2np(image)[0], cv2.COLOR_RGB2BGR)
    # fg_gt = cv2.cvtColor(tensor2np(fg_gt)[0], cv2.COLOR_RGB2BGR)
    bg_gt = cv2.cvtColor(tensor2np(bg_gt)[0], cv2.COLOR_RGB2BGR)
    mask_gt = tensor2np(mask_gt, isMask=True)[0]

    bg_pred,mask_preds = preds['bg'], preds['mask']
    # fg_pred = cv2.cvtColor(tensor2np(fg_pred)[0], cv2.COLOR_RGB2BGR)
    bg_pred = cv2.cvtColor(tensor2np(bg_pred)[0], cv2.COLOR_RGB2BGR)
    mask_preds = [tensor2np(m, isMask=True)[0] for m in mask_preds]
    main_mask = mask_preds[-2]
    mask_pred = mask_preds[0]
    outs = [image, bg_gt, bg_pred, mask_gt, mask_pred] #, main_mask]
    outimg = np.concatenate(outs, axis=1)
	
    if verbose==True:
        # print("show")
        cv2.imshow("out",outimg)
        cv2.waitKey(0)
    else:
        psnr = extra_infos['psnr']
        rmsew = extra_infos['rmsew']
        f1 = extra_infos['f1']

        img_fn = os.path.split(img_fn)[-1]
        out_fn = os.path.join(save_dir, "{}_psnr_{:.2f}_rmsew_{:.2f}_f1_{:.4f}{}".format(os.path.splitext(img_fn)[0],psnr,rmsew, f1, os.path.splitext(img_fn)[1]))
        cv2.imwrite(out_fn, outimg)





def main(args):
    args.dataset = args.dataset.lower()
    if args.dataset == 'clwd':
        dataset_func = datasets.CLWDDataset
    elif args.dataset == 'lvw':
        dataset_func = datasets.LVWDataset
    
    val_loader = torch.utils.data.DataLoader(dataset_func('test',args),batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    data_loaders = (None,val_loader)

    Machine = models.__dict__[args.models](datasets=data_loaders, args=args)

    
    model = Machine
    model.model.eval()
    print("==> testing VM model ")
    rmses = AverageMeter()
    rmsews = AverageMeter()
    ssimesx = AverageMeter()
    psnresx = AverageMeter()
    maskIoU = AverageMeter()
    maskF1 = AverageMeter()
    prime_maskIoU = AverageMeter()
    prime_maskF1 = AverageMeter()
    processTime = AverageMeter()

    prediction_dir = os.path.join(args.checkpoint,'rst')
    if not os.path.exists(prediction_dir): os.makedirs(prediction_dir)
    
    save_flag = False
    with torch.no_grad():
        for i, batches in enumerate(model.val_loader):

            inputs = batches['image'].to(model.device)
            target = batches['target'].to(model.device)
            mask =batches['mask'].to(model.device)
            wm =  batches['wm'].float().to(model.device)
            img_path = batches['img_path']

            # select the outputs by the giving arch
            start_time = time.time()
            outputs = model.model(model.norm(inputs))
            process_time = time.time() - start_time
            processTime.update((process_time*1000), inputs.size(0))

            imoutput,immask_all,imwatermark = outputs
            imoutput = imoutput[0] if is_dic(imoutput) else imoutput
            
            immask = immask_all[0]

            imfinal =imoutput*immask + model.norm(inputs)*(1-immask)
            psnrx = 10 * log10(1 / F.mse_loss(imfinal,target).item())       
            final_np = (imfinal.detach().cpu().numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
            target_np = (target.detach().cpu().numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
            # ssimx = ssim(final_np, target_np, multichannel=True)
            ssimx = pytorch_ssim.ssim(imfinal, target)
            
            
            
            rmsex = compute_RMSE(imfinal, target, mask, is_w=False)
            rmsewx = compute_RMSE(imfinal, target, mask, is_w=True)
            rmses.update(rmsex, inputs.size(0))
            rmsews.update(rmsewx, inputs.size(0))
            psnresx.update(psnrx, inputs.size(0))
            ssimesx.update(ssimx, inputs.size(0))


            main_mask = immask_all[1::2]
            comp_mask = immask_all[2::2]
            out_mask = main_mask[-1]
            comp_mask = comp_mask[-1]
            
            comp_sets = []
            prime_mask_pred = torch.where(out_mask > 0.5, torch.ones_like(out_mask), torch.zeros_like(out_mask)).to(out_mask.device)
            mask_pred = torch.where(comp_mask > 0.5, torch.ones_like(out_mask), torch.zeros_like(out_mask)).to(out_mask.device)
           
            iou = compute_IoU(prime_mask_pred, mask)
            prime_maskIoU.update(iou)
            f1 = FScore(prime_mask_pred, mask).item()
            prime_maskF1.update(f1, inputs.size(0))

            iou = compute_IoU(mask_pred, mask)
            maskIoU.update(iou)
            f1 = FScore(mask_pred, mask).item()
            maskF1.update(f1, inputs.size(0))

            if save_flag:
                save_output(
                    inputs={'I':inputs, 'bg':target,  'mask':mask}, 
                    preds={'bg':imfinal, 'mask':immask_all}, 
                    save_dir=prediction_dir, 
                    img_fn=img_path[0], 
                    extra_infos={"psnr":psnrx, "rmsew":rmsewx, "f1":f1},
                    verbose=False
                )
            if i % 100 == 0:
                print("Batch[%d/%d]| PSNR:%.4f | SSIM:%.4f | RMSE:%.4f | RMSEw:%.4f | primeIoU:%.4f, primeF1:%.4f | maskIoU:%.4f | maskF1:%.4f | time:%.2f"
                %(i,len(model.val_loader),psnresx.avg,ssimesx.avg, rmses.avg, rmsews.avg, prime_maskIoU.avg, prime_maskF1.avg, maskIoU.avg, maskF1.avg, processTime.avg))
    print("Total:\nPSNR:%.4f | SSIM:%.4f | RMSE:%.4f | RMSEw:%.4f | primeIoU:%.4f, primeF1:%.4f | maskIoU:%.4f | maskF1:%.4f | time:%.2f"
                %(psnresx.avg,ssimesx.avg, rmses.avg, rmsews.avg, prime_maskIoU.avg, prime_maskF1.avg, maskIoU.avg, maskF1.avg, processTime.avg))
    print("DONE.\n")


if __name__ == '__main__':
    parser=Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
    main(parser.parse_args())
    
