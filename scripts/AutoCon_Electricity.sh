export CUDA_VISIBLE_DEVICES=$1

nitr=$2

python -u run.py --AutoCon  --AutoCon_multiscales 96 --AutoCon_wnorm Mean  --AutoCon_lambda 0.001  --d_model 4 --d_ff 4 --e_layers 1 --target OT --c_out 1 --root_path ./dataset/electricity --data_path electricity.csv --model_id ICLR24_CRV --model AutoConNet --data electricity --seq_len 336 --label_len 48 --pred_len 96 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.001 --feature S
python -u run.py --AutoCon  --AutoCon_multiscales 336 --AutoCon_wnorm Mean  --AutoCon_lambda 0.001  --d_model 8 --d_ff 8 --e_layers 2 --target OT --c_out 1 --root_path ./dataset/electricity --data_path electricity.csv --model_id ICLR24_CRV --model AutoConNet --data electricity --seq_len 336 --label_len 48 --pred_len 720 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.001 --feature S
python -u run.py --AutoCon  --AutoCon_multiscales 336 --AutoCon_wnorm Mean  --AutoCon_lambda 0.001  --d_model 16 --d_ff 16 --e_layers 2 --target OT --c_out 1 --root_path ./dataset/electricity --data_path electricity.csv --model_id ICLR24_CRV --model AutoConNet --data electricity --seq_len 336 --label_len 48 --pred_len 1440 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.001 --feature S
python -u run.py --AutoCon  --AutoCon_multiscales 96 --AutoCon_wnorm Mean  --AutoCon_lambda 0.001  --d_model 16 --d_ff 16 --e_layers 2 --target OT --c_out 1 --root_path ./dataset/electricity --data_path electricity.csv --model_id ICLR24_CRV --model AutoConNet --data electricity --seq_len 168 --label_len 48 --pred_len 2160 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.005 --feature S