python tools/evaluate.py \
  --gpu=3 \
  --input_height=416 \
  --input_width=416 \
  --dataset_val_path="./data/voc_0712_val.txt" \
  --dataset_format="yolo" \
  --num_classes=20 \
  --classes_info_file="./data/voc.txt" \
  --outputs_dir="./outputs/yolo_voc/" \
  --test_weight="./logs/yolo_voc/weights/epoch=149_loss=0.5854_val_loss=3.3467.pt"