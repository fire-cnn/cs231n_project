#!/usr/bin/bash

echo "Running CLIP saved model"

python run_clip.py \
    --output_dir ./vit-roberta-finetuned \
    --model_name_or_path ./vit-roberta \
    --data_dir ./data \
    --train_file ./train_data.csv \
    --validation_file ./test_data.csv \
    --image_column image_column \
    --caption_column caption_column \
    --remove_unused_columns=False \
    --do_train --do_eval \
    --num_train_epochs="10" \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
    --learning_rate="5e-5" --warmup_steps="500" --weight_decay 0.1 \
    --overwrite_output_dir
