import argparse
import torch
import os
import cv2
import numpy as np

torch.backends.cudnn.benchmark = True

import datasets as datasets
import src.models as models
from options import Options
import torch.nn.functional as F



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
    image = inputs['I'] #, inputs['bg'], inputs['mask']
    image = cv2.cvtColor(tensor2np(image)[0], cv2.COLOR_RGB2BGR)

    bg_pred,mask_preds = preds['bg'], preds['mask']
    bg_pred = cv2.cvtColor(tensor2np(bg_pred)[0], cv2.COLOR_RGB2BGR)
    mask_pred = tensor2np(mask_preds, isMask=True)[0]
    outs = [image, bg_pred,  mask_pred]
    outimg = np.concatenate(outs, axis=1)
	
    if verbose==True:
        # print("show")
        cv2.imshow("out",outimg)
        cv2.waitKey(0)
    else:
        img_fn = os.path.split(img_fn)[-1]
        out_fn = os.path.join(save_dir, "{}{}".format(os.path.splitext(img_fn)[0], os.path.splitext(img_fn)[1]))
        cv2.imwrite(out_fn, outimg)

def preprocess(file_path, img_size=512):
    img_J = cv2.imread(file_path)
    assert img_J is not None, "NoneType"
    h,w,_ = img_J.shape
    img_J = cv2.cvtColor(img_J, cv2.COLOR_BGR2RGB).astype(np.float)/255.
    img_J = torch.from_numpy(img_J.transpose(2,0,1)[np.newaxis,...]) #[1,C,H,W]
    img_J = F.interpolate(img_J, size=(img_size, img_size), mode='bilinear')
    
    return img_J


def test_dataloder(img_path, crop_size):
    loaders = []
    save_fns = []

    for root, dirs, fns in os.walk(img_path):
        for dir in dirs:
            path = os.path.join(root, dir)
            fn_list = os.listdir(path)
            for fn in fn_list:
                if fn.startswith('.'): continue
                if not (fn.endswith('.jpg') or fn.endswith('jpeg') or fn.endswith('png') ): continue
                fn = os.path.join(path, fn)
                J = preprocess(fn, img_size=crop_size)
                loaders.append(J)
                save_fns.append(fn)
    return loaders,save_fns



def main(args):

    Machine = models.__dict__[args.models](datasets=(None, None), args=args)

    model = Machine
    model.model.eval()
    print("==> testing VM model ")

    prediction_dir = os.path.join(args.test_dir,'rst')
    if not os.path.exists(prediction_dir): os.makedirs(prediction_dir)
    
    doc_loader,fns = test_dataloder(args.test_dir, args.crop_size)
    with torch.no_grad():
        for i, batches in enumerate(zip(doc_loader, fns)):
            inputs, fn = batches[0], batches[1]
            inputs = inputs.to(model.device).float()
            outputs = model.model(inputs)
            imoutput,immask_all,imwatermark = outputs

            imoutput = imoutput[0]  
            immask = immask_all[0]

            imfinal =imoutput*immask + model.norm(inputs)*(1-immask)
            save_output(
                inputs = {'I':inputs},
                preds = {'bg':imfinal, 'mask':immask},
                save_dir= prediction_dir,
                img_fn = fn
            )
            
            




if __name__ == '__main__':
    parser=Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
    main(parser.parse_args())
   
