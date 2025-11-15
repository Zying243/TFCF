export CUDA_VISIBLE_DEVICES=0

model_name=TexFilter

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --enc_in 21 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --embed_size 128 \
  --hidden_size 128 \
  --dropout 0 \
  --train_epochs 20 \
  --batch_size 128 \
  --patience 6 \
  --learning_rate 0.01 \
  --des 'Exp' \
  --itr 1
