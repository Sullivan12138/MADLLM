python -u run.py \
  --is_training 1 \
  --root_path ./all_datasets/SMAP \
  --model GPT4TS \
  --model_name SMAP \
  --data SMAP \
  --seq_len 100 \
  --gpt_layer 6 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 1 \
  --stride 1 \
  --enc_in 25 \
  --c_out 25 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --learning_rate 0.0005 \
  --train_epochs 5 \
  --channels 38 \
  --use_skip_embedding 1 \
  --use_prompt_pool 1 \
  --use_feature_embedding 1 \
  --nb_random_samples 10 \
  --prompt_len 5 \
  --top_k 5 \
  --pool_size 10