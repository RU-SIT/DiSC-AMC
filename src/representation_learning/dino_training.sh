#!/bin/bash

# Training script for DINO autoencoder and classifier

# --- Configuration ---
num_workers=10
batch_size=128
eval_step=5

# # --- Train Autoencoder ---
# echo "--- Training DINO Autoencoder ---"
# python autoencoder_training.py \
#     --model dino \
#     --save_path ../../exp/dino_autoencoder.pth \
#     --num_epochs 50 \
#     --learning_rate 1e-4 \
#     --num_workers $num_workers \
#     --eval_step $eval_step \
#     --batch_size $batch_size

# --- Train Classifier ---
echo "--- Training DINO Classifier ---"
python classifier_training.py \
    --model dino \
    --pretrained_path ../../exp/dino_autoencoder.pth \
    --save_path ../../exp/dino_classifier.pth \
    --num_epochs 50 \
    --learning_rate 5e-4 \
    --num_workers $num_workers \
    --eval_step $eval_step \
    --batch_size $batch_size