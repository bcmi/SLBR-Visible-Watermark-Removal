import torch
import torch.nn as nn
from progress.bar import Bar
from tqdm import tqdm
import pytorch_ssim
import json
import sys,time,os
import torchvision
from math import log10
import numpy as np
from .BasicModel import BasicModel
from evaluation import AverageMeter, compute_IoU, FScore, compute_RMSE
import torch.nn.functional as F
from src.utils.parallel import DataParallelModel, DataParallelCriterion
from src.utils.losses import VGGLoss, l1_relative,is_dic
from src.utils.imutils import im_to_numpy
import skimage.io
from skimage.measure import compare_psnr,compare_ssim
import torchvision
import pytorch_iou
import pytorch_ssim

class Losses(nn.Module):
    def __init__(self, argx, device, norm_func, denorm_func):
        super(Losses, self).__init__()
        self.args = argx
        self.masked_l1_loss, self.mask_loss = l1_relative, nn.BCELoss()
        self.l1_loss = nn.L1Loss()

        if self.args.lambda_content > 0:
            self.vgg_loss = VGGLoss(self.args.sltype, style=self.args.lambda_style>0).to(device)
        
        if self.args.lambda_iou > 0:
            self.iou_loss = pytorch_iou.IOU(size_average=True)

        self.lambda_primary = self.args.lambda_primary
        self.gamma = 0.5
        self.norm = norm_func
        self.denorm = denorm_func

    def forward(self, synthesis, pred_ims, target, pred_ms, mask, threshold=0.5):
        pixel_loss, refine_loss, vgg_loss, mask_loss = [0]*4
        pred_ims = pred_ims if is_dic(pred_ims) else [pred_ims]
        
        # reconstruction loss
        pixel_loss += self.masked_l1_loss(pred_ims[-1], target, mask) # coarse stage
        if len(pred_ims) > 1:
            refine_loss = self.masked_l1_loss(pred_ims[0], target, mask) # refinement stage
        
        recov_imgs = [ self.denorm(pred_im*mask + (1-mask)*self.norm(target)) for pred_im in pred_ims ]        
        pixel_loss += sum([self.l1_loss(im,target) for im in recov_imgs]) * 1.5
        

        if self.args.lambda_content > 0:
            vgg_loss = [self.vgg_loss(im,target,mask) for im in recov_imgs]
            vgg_loss = sum([vgg['content'] for vgg in vgg_loss]) * self.args.lambda_content + \
                       sum([vgg['style'] for vgg in vgg_loss]) * self.args.lambda_style

        # mask loss
        pred_ms = [F.interpolate(ms, size=mask.shape[2:], mode='bilinear') for ms in pred_ms]
        pred_ms = [pred_m.clamp(0,1) for pred_m in pred_ms]
        mask = mask.clamp(0,1)

        final_mask_loss = 0
        final_mask_loss += self.mask_loss(pred_ms[0], mask)
        
        primary_mask = pred_ms[1::2][::-1]
        self_calibrated_mask = pred_ms[2::2][::-1]
        # primary prediction
        primary_loss =  sum([self.mask_loss(pred_m, mask) * (self.gamma**i) for i,pred_m in enumerate(primary_mask)])
        # self calibrated Branch
        self_calibrated_loss =  sum([self.mask_loss(pred_m, mask) * (self.gamma**i) for i,pred_m in enumerate(self_calibrated_mask)])
        if self.args.lambda_iou > 0:
            self_calibrated_loss += sum([self.iou_loss(pred_m, mask) * (self.gamma**i) for i,pred_m in enumerate(self_calibrated_mask)]) * self.args.lambda_iou

        mask_loss = final_mask_loss + self_calibrated_loss + self.lambda_primary * primary_loss
        return pixel_loss, refine_loss, vgg_loss, mask_loss




class SLBR(BasicModel):
    def __init__(self,**kwargs):
        BasicModel.__init__(self,**kwargs)
        self.loss = Losses(self.args, self.device, self.norm, self.denorm)
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        self.model.set_optimizers()
        if self.args.resume != '':
            self.resume(self.args.resume)
       
    def train(self,epoch):

        self.current_epoch = epoch

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_meter = AverageMeter()
        loss_mask_meter = AverageMeter()
        loss_vgg_meter = AverageMeter()
        loss_refine_meter = AverageMeter()
        f1_meter = AverageMeter()
        # switch to train mode
        self.model.train()

        end = time.time()
        bar = Bar('Processing {} '.format(self.args.nets), max=len(self.train_loader))
        for i, batches in enumerate(self.train_loader):
            current_index = len(self.train_loader) * epoch + i

            inputs = batches['image'].float().to(self.device)
            target = batches['target'].float().to(self.device)
            mask = batches['mask'].float().to(self.device)
            # wm =  batches['wm'].float().to(self.device)
            # alpha_gt = batches['alpha'].float().to(self.device)
            img_path = batches['img_path']
            
            outputs = self.model(self.norm(inputs))
            self.model.zero_grad_all()
            coarse_loss, refine_loss, style_loss, mask_loss = self.loss(
                inputs,outputs[0],self.norm(target),outputs[1],mask)
            
            total_loss = self.args.lambda_l1*(coarse_loss+refine_loss) + self.args.lambda_mask * (mask_loss)  + style_loss
            
            # compute gradient and do SGD step
            total_loss.backward()
            self.model.step_all()

            # measure accuracy and record loss
            losses_meter.update(coarse_loss.item(), inputs.size(0))
            loss_mask_meter.update(mask_loss.item(), inputs.size(0))
            if isinstance(refine_loss,int):
                loss_refine_meter.update(refine_loss, inputs.size(0))
            else:
                loss_refine_meter.update(refine_loss.item(), inputs.size(0))
            
            f1 = FScore(outputs[1][0], mask).item()
            f1_meter.update(f1, inputs.size(0))
            if self.args.lambda_content > 0  and not isinstance(style_loss,int):
                loss_vgg_meter.update(style_loss.item(), inputs.size(0))

            # measure elapsed timec
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            suffix  = "({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | loss L1: {loss_label:.4f} | loss Refine: {loss_refine:.4f} | loss VGG: {loss_vgg:.4f} | loss Mask: {loss_mask:.4f} | mask F1: {mask_f1:.4f}".format(
                        batch=i + 1,
                        size=len(self.train_loader),
                        data=data_time.val,
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss_label=losses_meter.avg,
                        loss_refine=loss_refine_meter.avg,
                        loss_vgg=loss_vgg_meter.avg,
                        loss_mask=loss_mask_meter.avg,
                        mask_f1=f1_meter.avg,
                        )
            if current_index % 100 == 0:
                print(suffix)

            if self.args.freq > 0 and current_index % self.args.freq == 0:
                self.validate(current_index)
                self.flush()
                self.save_checkpoint()
            if i % 100 == 0:
                self.record('train/loss_L2', losses_meter.avg, current_index)
                self.record('train/loss_Refine', loss_refine_meter.avg, current_index)
                self.record('train/loss_VGG', loss_vgg_meter.avg, current_index)
                self.record('train/loss_Mask', loss_mask_meter.avg, current_index)
                self.record('train/mask_F1', f1_meter.avg, current_index)

                mask_pred = outputs[1][0]
                bg_pred = self.denorm(outputs[0][0]*mask_pred + (1-mask_pred)*self.norm(inputs))
                show_size = 5 if inputs.shape[0] > 5 else inputs.shape[0]
                self.image_display = torch.cat([
                    inputs[0:show_size].detach().cpu(),             # input image
                    target[0:show_size].detach().cpu(),                        # ground truth
                    bg_pred[0:show_size].detach().cpu(),       # refine out
                    mask[0:show_size].detach().cpu().repeat(1,3,1,1),
                    outputs[1][0][0:show_size].detach().cpu().repeat(1,3,1,1),
                    outputs[1][-2][0:show_size].detach().cpu().repeat(1,3,1,1)
                ],dim=0)
                image_dis = torchvision.utils.make_grid(self.image_display, nrow=show_size)
                self.writer.add_image('Image', image_dis, current_index)
            del outputs


    def validate(self, epoch):

        self.current_epoch = epoch
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_meter = AverageMeter()
        loss_mask_meter = AverageMeter()
        psnr_meter = AverageMeter()
        fpsnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        rmse_meter = AverageMeter()
        rmsew_meter = AverageMeter()
        

        coarse_psnr_meter = AverageMeter()
        coarse_rmsew_meter = AverageMeter()

        iou_meter = AverageMeter()
        f1_meter = AverageMeter()
        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        bar = Bar('Processing {} '.format(self.args.nets), max=len(self.val_loader))
        with torch.no_grad():
            for i, batches in enumerate(self.val_loader):
                
                current_index = len(self.val_loader) * epoch + i

                inputs = batches['image'].to(self.device)
                target = batches['target'].to(self.device)
                mask = batches['mask'].to(self.device)
                # alpha_gt = batches['alpha'].float().to(self.device)

                outputs = self.model(self.norm(inputs))
                imoutput,immask,imwatermark = outputs
                
                immask = immask[0]
                if len(imoutput) > 1:
                    imcoarse = imoutput[1]
                    imcoarse = imcoarse*immask + inputs*(1-immask)
                else: imcoarse = None
                imoutput = imoutput[0] if is_dic(imoutput) else imoutput

                imfinal = self.denorm(imoutput*immask + self.norm(inputs)*(1-immask))

                eps = 1e-6
                psnr = 10 * log10(1 / F.mse_loss(imfinal,target).item()) 
                fmse = F.mse_loss(imfinal*mask, target*mask, reduction='none').sum(dim=[1,2,3]) / (mask.sum(dim=[1,2,3])*3+eps)
                fpsnr = 10 * torch.log10(1 / fmse).mean().item()
                ssim = pytorch_ssim.ssim(imfinal,target)
                if imcoarse is not None:
                    psnr_coarse = 10 * log10(1 / F.mse_loss(imcoarse,target).item())  
                    rmsew_coarse = compute_RMSE(imcoarse, target, mask, is_w=True)
                    coarse_psnr_meter.update(psnr_coarse, inputs.size(0))
                    coarse_rmsew_meter.update(rmsew_coarse, inputs.size(0))

                psnr_meter.update(psnr, inputs.size(0))
                fpsnr_meter.update(fpsnr, inputs.size(0))
                ssim_meter.update(ssim, inputs.size(0))
                rmse_meter.update(compute_RMSE(imfinal,target,mask),inputs.size(0))
                rmsew_meter.update(compute_RMSE(imfinal,target,mask,is_w=True), inputs.size(0))

                iou = compute_IoU(immask, mask)
                iou_meter.update(iou, inputs.size(0))
                f1 = FScore(immask, mask).item()
                f1_meter.update(f1, inputs.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                if imcoarse is None:
                    suffix  = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | PSNR: {psnr:.4f} | fPSNR: {fpsnr:.4f} | SSIM: {ssim:.4f} | RMSE: {rmse:.4f} | RMSEw: {rmsew:.4f} | IoU: {iou:.4f} | F1: {f1:.4f}'.format(
                                batch=i + 1,
                                size=len(self.val_loader),
                                data=data_time.val,
                                bt=batch_time.val,
                                total=bar.elapsed_td,
                                eta=bar.eta_td,
                                psnr=psnr_meter.avg,
                                fpsnr=fpsnr_meter.avg,
                                ssim=ssim_meter.avg,
                                rmse=rmse_meter.avg,
                                rmsew=rmsew_meter.avg,
                                iou=iou_meter.avg,
                                f1=f1_meter.avg
                                )
                else:
                    suffix  = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | CPSNR: {cpsnr:.4f} | CRMSEw: {crmsew:.4f} | PSNR: {psnr:.4f} | fPSNR: {fpsnr:.4f} | RMSE: {rmse:.4f} | RMSEw: {rmsew:.4f} | SSIM: {ssim:.4f} | IoU: {iou:.4f} | F1: {f1:.4f}'.format(
                            batch=i + 1,
                            size=len(self.val_loader),
                            data=data_time.val,
                            bt=batch_time.val,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            cpsnr=coarse_psnr_meter.avg,
                            crmsew=coarse_rmsew_meter.avg,
                            psnr=psnr_meter.avg,
                            fpsnr=fpsnr_meter.avg,
                            ssim=ssim_meter.avg,
                            rmse=rmse_meter.avg,
                            rmsew=rmsew_meter.avg,
                            iou=iou_meter.avg,
                            f1=f1_meter.avg
                            )
                if i%100 == 0:
                    print(suffix)
                # bar.next()
        print("Total:")
        print(suffix)
        bar.finish()
        
        print("Iter:%s,losses:%s,PSNR:%.4f,SSIM:%.4f"%(epoch, losses_meter.avg,psnr_meter.avg,ssim_meter.avg))
        self.record('val/loss_L2', losses_meter.avg, epoch)
        self.record('val/loss_mask', loss_mask_meter.avg, epoch)
        self.record('val/PSNR', psnr_meter.avg, epoch)
        self.record('val/SSIM', ssim_meter.avg, epoch)
        self.record('val/RMSEw', rmsew_meter.avg, epoch)
        self.metric = psnr_meter.avg

        self.model.train()

   