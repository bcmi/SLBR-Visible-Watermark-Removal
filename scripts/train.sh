K_CENTER=1
K_REFINE=3
K_SKIP=3
VGG_LOSS=2.5e-2
IOU_LOSS=0 #0.5
PRIMARY_LOSS=0.01
INPUT_SIZE=256
DATASET=CLWD
NAME=slbr_K${K_CENTER}_VGG${VGG_LOSS}_${IOU_LOSS}_${DATASET}_${INPUT_SIZE}
# nohup python -u   main.py \
python -m pdb train.py \
 --epochs 100 \
 --schedule 100 \
 --lr 1e-3 \
 --gpu_id 1 \
 --checkpoint /media/sda/Watermark \
 --dataset_dir /media/sda/datasets/Watermark/${DATASET} \
 --nets slbr  \
 --sltype vggx \
 --lambda_style ${VGG_LOSS} \
 --lambda_iou ${IOU_LOSS} \
 --lambda_primary ${PRIMARY_LOSS} \
 --masked True \
 --loss-type hybrid \
 --models slbr \
  --input-size ${INPUT_SIZE} \
 --crop_size ${INPUT_SIZE} \
 --train-batch 2 \
 --test-batch 1 \
 --preprocess resize_and_crop \
 --name ${NAME} \
 --k_center ${K_CENTER} \
 --dataset ${DATASET} \
 --use_refine \
 --k_refine ${K_REFINE} \
 --k_skip_stage ${K_SKIP}

#  --resume /media/sda/Watermark/SplitNet_${NAME}/model_best.pth.tar
