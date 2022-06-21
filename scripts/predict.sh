python tools/predict.py \
  --gpu=0 \
  --input_height=416 \
  --input_width=416 \
  --num_classes=20 \
  --classes_info_file="./data/voc.txt" \
  --outputs_dir="./outputs/yolo_voc" \
  --test_weight="./logs/yolo_voc/weights/epoch=149_loss=0.5854_val_loss=3.3467.pt"