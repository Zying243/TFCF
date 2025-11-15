export CUDA_VISIBLE_DEVICES=2

model_name=TexFilter

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --enc_in 321 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --embed_size 512 \
  --hidden_size 512 \
  --dropout 0 \
  --train_epochs 20 \
  --batch_size 4 \
  --patience 6 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1
