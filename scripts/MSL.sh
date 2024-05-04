python -u run.py \
  --is_training 1 \
  --root_path ./all_datasets/MSL \
  --model GPT4TS \
  --data MSL \
  --seq_len 100 \
  --gpt_layer 6 \
  --d_model 768 \
  --d_ff 8 \
  --patch_size 1 \
  --stride 1 \
  --enc_in 55 \
  --c_out 55 \
  --anomaly_ratio 2 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --feature_epochs 10 \
  --feature_lr 0.001 \
  --use_skip_embedding 1 \
  --use_prompt_pool 1 \
  --use_feature_embedding 1 \
  --nb_random_samples 10 \
  --top_k 10 \
  --pool_size 20 \
  --prompt_len 20
  