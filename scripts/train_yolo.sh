python tools/train.py \
  --gpu=0,1,2,3 \
  --input_height=416 \
  --input_width=416 \
  --dataset_train_path="./data/voc_0712_train.txt" \
  --dataset_val_path="./data/voc_0712_val.txt" \
  --dataset_format="yolo" \
  --num_classes=20 \
  --classes_info_file="./data/voc.txt" \
  --num_workers=8 \
  --pretrain_weight_path="resnet50.pth" \
  --learn_rate_init=5e-4 \
  --learn_rate_end=5e-6 \
  --warmup_epochs=5 \
  --freeze_epochs=50 \
  --unfreeze_epochs=100 \
  --freeze_batch_size=160 \
  --unfreeze_batch_size=80 \
  --logs_dir="./logs/yolo_voc_resnet50"