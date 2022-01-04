K_CENTER=2
K_REFINE=3
K_SKIP=3
MASK_MODE=res #'cat'

L1_LOSS=2
CONTENT_LOSS=2.5e-1
STYLE_LOSS=2.5e-1
PRIMARY_LOSS=0.01
IOU_LOSS=0.25 

INPUT_SIZE=256
DATASET=CLWD
NAME=slbr_v1
# nohup python -u   main.py \
CUDA_VISIBLE_DEVICES=1 python -u train.py \
 --epochs 100 \
 --schedule 65 \
 --lr 1e-3 \
 --gpu_id 1 \
 --checkpoint /media/sda/Watermark \
 --dataset_dir /media/sda/datasets/Watermark/${DATASET} \
 --nets slbr  \
 --sltype vggx \
 --mask_mode ${MASK_MODE} \
 --lambda_content ${CONTENT_LOSS} \
 --lambda_style ${STYLE_LOSS} \
 --lambda_iou ${IOU_LOSS} \
 --lambda_l1 ${L1_LOSS} \
 --lambda_primary ${PRIMARY_LOSS} \
 --masked True \
 --loss-type hybrid \
 --models slbr \
  --input-size ${INPUT_SIZE} \
 --crop_size ${INPUT_SIZE} \
 --train-batch 8 \
 --test-batch 1 \
 --preprocess resize \
 --name ${NAME} \
 --k_center ${K_CENTER} \
 --dataset ${DATASET} \
 --use_refine \
 --k_refine ${K_REFINE} \
 --k_skip_stage ${K_SKIP} \
#  --start-epoch 70 \
#  --resume /media/sda/Watermark/${NAME}/model_best.pth.tar
