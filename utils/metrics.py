import numpy as np
import torch
from utils.dtwloss.dilate_loss import dilate_loss

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def shape_metric(pred, true, batch_size=100):
    dilate_e = 0.0
    shape_e = 0.0
    temp_e = 0.0
    pred = torch.tensor(pred).cuda()
    true = torch.tensor(true).cuda()
    for st in range(0, pred.shape[0], batch_size):
        st = st
        ed = st + batch_size
        # print(st, ed)
        sub_pred = pred[st:ed]
        sub_true = true[st:ed]
        n_data = sub_true.shape[0]
        with torch.no_grad():
            s_dilate_e, s_shape_DTW, s_temporal_DTW = dilate_loss(sub_pred, sub_true, 0.5, 0.01, 'cuda')

            dilate_e += s_dilate_e.cpu().item() * n_data
            shape_e += s_shape_DTW.cpu().item() * n_data
            temp_e += s_temporal_DTW.cpu().item() * n_data

    return dilate_e/pred.shape[0], shape_e/pred.shape[0], temp_e/pred.shape[0]


