export CUDA_VISIBLE_DEVICES=$1

nitr=$2

python -u run.py --AutoCon  --AutoCon_multiscales 96  --AutoCon_wnorm LastVal  --AutoCon_lambda 1.0  --d_model 16 --d_ff 16 --e_layers 2 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTh1.csv --model_id ICLR24_CRV --model AutoConNet --data ETTh1 --seq_len 96 --label_len 48 --pred_len 96 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.01 --feature S
python -u run.py --AutoCon  --AutoCon_multiscales 720  --AutoCon_wnorm LastVal  --AutoCon_lambda 1.0  --d_model 16 --d_ff 16 --e_layers 2 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTh1.csv --model_id ICLR24_CRV --model AutoConNet --data ETTh1 --seq_len 96 --label_len 48 --pred_len 720 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.005 --feature S
python -u run.py --AutoCon  --AutoCon_multiscales 1440  --AutoCon_wnorm LastVal  --AutoCon_lambda 1.0  --d_model 16 --d_ff 16 --e_layers 1 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTh1.csv --model_id ICLR24_CRV --model AutoConNet --data ETTh1 --seq_len 96 --label_len 48 --pred_len 1440 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.005 --feature S
python -u run.py --AutoCon  --AutoCon_multiscales 1440  --AutoCon_wnorm LastVal  --AutoCon_lambda 1.0  --d_model 16 --d_ff 32 --e_layers 1 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTh1.csv --model_id ICLR24_CRV --model AutoConNet --data ETTh1 --seq_len 96 --label_len 48 --pred_len 2160 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.005 --feature S