# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=0

python -u main.py \
  --train_ratio=1.0 \
  --valid_ratio=1.0 \
  --embedding_model="glove" \
  --multi_path \
  --min_count=1 \
  --is_mask \
  --all_layer_output \
  --output_linear_combine \
  --child_lr_decay_scheme="cosine" \
  --is_cuda \
  --search_for="macro" \
  --reset_output_dir \
  --data_path="./data/sst" \
  --embedding_path="./data/glove.840B.300d.txt" \
  --class_num=5 \
  --child_optim_algo="momentum" \
  --child_optim_algo="adam" \
  --max_input_length=32 \
  --output_dir="nas_outputs" \
  --batch_size=128 \
  --num_epochs=150 \
  --log_every=50 \
  --eval_every_epochs=1 \
  --child_num_layers=24 \
  --child_out_filters=32 \
  --child_l2_reg=0.00002 \
  --child_num_branches=8 \
  --child_grad_bound=5.0 \
  --cnn_keep_prob=0.8 \
  --final_output_keep_prob=1.0 \
  --embed_keep_prob=0.8 \
  --lstm_out_keep_prob=0.8 \
  --attention_keep_prob=0.8 \
  --child_lr=0.005 \
  --child_lr_max=0.005 \
  --child_lr_min=0.0001 \
  --child_lr_T_0=10 \
  --child_lr_T_mul=2 \
  --controller_training \
  --controller_entropy_weight=0.0001 \
  --controller_train_every=1 \
  --controller_train_steps=20 \
  --controller_lr=0.001 \
  --controller_tanh_constant=1.5 \
  --controller_skip_target=0.4 \
  --controller_skip_weight=0.8 \
  "$@"
