# Self-Supervised-Contrastive-Forecasting

This is an official repository for the paper, titled [Self-Supervised-Contrastive-Forecasting](https://openreview.net/forum?id=nBCuRzjqK7), accepted at ICLR'24.

Our repo includes Autocorrelation-based Contrastive Loss (AutoCon) and Model Architecture (AutoConNet).

Our code is implemented based on [TSLib](https://github.com/thuml/Time-Series-Library), and it also includes some time series models. If you want to evaluate more models, please check the TSLib. We sincerely thank the creators of TSLib for their contributions to the time series community.

## Environment setup
```
python == 3.8
torch == 1.7.1
numpy == 1.23.5
pandas
statsmodels
scikit-learn
einops
sympy
numba
```

## Run with Command Line 
```
python run.py --AutoCon --AutoCon_multiscales 720 --AutoCon_wnorm LastVal --AutoCon_lambda 0.1 --d_model 16 --d_ff 16 --e_layers 3 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTh2.csv --model_id ICLR24 --model AutoConNet --data ETTh2 --seq_len 336 --pred_len 720 --enc_in 1 --des 'Exp' --itr 1 --batch_size 64 --learning_rate 0.005 --feature S
```

## Run with Scripts
sh ./scripts/AutoCon_{ETTh1|ETTh2|ETTm1|ETTm2|Electricity|Traffic|Weather|Excange|Illness} {CUDA_VISBLE_DEVICES} {# OF RUNS}

Examples
```
$pwd
/home/user/Self-Supervised-Contrastive-Forecasting

$sh ./scripts/AutoCon_ETTh2.sh 0 5 
$sh ./scripts/AutoCon_Traffic.sh 0 5
```

## Reproducibility
For reproducibility, we provide [logs](https://github.com/junwoopark92/Self-Supervised-Contrastive-Forecsating/tree/main/reproducibility/Table1-Extended-long-term-forecasting) of the experimental results in the paper. These logs are generated from the evaluation of the same model five times. We observed performance variations based on GPU devices due to CUDA optimization for CNN layers even though a fixed seed. Our experiments were primarily conducted on the RTX3090.

## Citations
```
@inproceedings{park2024self,
  title={Self-Supervised Contrastive Forecasting},
  author={Junwoo Park and Daehoon Gwak and Jaegul Choo and Edward Choi},
  booktitle={Proc. the International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
