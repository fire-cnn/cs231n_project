#!/usr/bin/bash

echo "Running CLIP saved model"

python run_clip.py \
    --model_name_or_path ./vit-roberta \
    --output_dir /scratch/users/ihigueme/vit-roberta-finetuned-supervised-very-small-k \
    --data_dir  /oak/stanford/groups/mburke/cnn_fire_fuel/data \
    --train_file /oak/stanford/groups/mburke/cnn_fire_fuel/data/text/train_data.csv \
    --validation_file /oak/stanford/groups/mburke/cnn_fire_fuel/data/text/test_data.csv \
    --test_file /oak/stanford/groups/mburke/cnn_fire_fuel/data/text/test_data.csv \
    --image_column image_column \
    --caption_column caption_column \
    --remove_unused_columns=False \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy epoch \
    --max_seq_length 70 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate="5e-5" \
    --warmup_steps 500 \
    --weight_decay 0.001 \
    --overwrite_output_dir \
    --save_strategy epoch \ 
    --max_train_samples 10 \
    --max_eval_samples 10 
