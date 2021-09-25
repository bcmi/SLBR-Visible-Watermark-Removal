K_CENTER=1
K_REFINE=3
K_SKIP=3
VGG_LOSS=2.5e-2
IOU_LOSS=0 #0.5
PRIMARY_LOSS=0.01
INPUT_SIZE=256
DATASET=CLWD
NAME=slbr_K${K_CENTER}_VGG${VGG_LOSS}_${IOU_LOSS}_${DATASET}_${INPUT_SIZE}

CUDA_VISIBLE_DEVICES=0 python3  test.py \
  -c /media/sda/Watermark/SplitNet \
  --nets slbr \
  --models slbr \
  --input-size ${INPUT_SIZE} \
  --crop_size ${INPUT_SIZE} \
  --test-batch 1 \
  --evaluate\
  --dataset_dir /media/sda/datasets/Watermark/${DATASET} \
  --preprocess resize \
  --no_flip \
  --name ${NAME} \
  --k_center ${K_CENTER} \
  --use_refine \
  --k_refine ${K_REFINE} \
  --k_skip_stage ${K_SKIP}
  --dataset ${DATASET} \
  --resume /media/sda/Watermark/${NAME}/model_best.pth.tar \
