from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, shape_metric
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

from statsmodels.tsa.api import acf
from models.AutoConNet import series_decomp
from layers.losses import AutoCon, AutoConCI

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_with_AutoCon(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_with_AutoCon, self).__init__(args)

        self.AutoCon = args.AutoCon
        self.AutoCon_lambda = args.AutoCon_lambda
        st = time.time()
        self.AutoCon_loss = self.init_AutoCon(args)
        ed = time.time()
        print(f'Autocorrelation calculation time: {ed-st:.4f}')

    def init_AutoCon(self, args):
        target_data, _ = self._get_data(flag='train')
        target_data = target_data.data_x.copy()
        smoother = series_decomp(args.seq_len+1)
        x = torch.from_numpy(target_data).unsqueeze(0)
        _, target_data = smoother(x)
        target_data = target_data.squeeze(0).numpy()
        acf_values = []
        for i_ch in range(target_data.shape[-1]):
            acf_values.append(acf(target_data[..., i_ch], nlags=len(target_data)))

        if self.args.model == 'AutoConCI':
            acf_values = np.stack(acf_values, axis=0)
            loss = AutoConCI(args.batch_size, args.seq_len, np.abs(acf_values), temperature=1.0, base_temperature=1.0)
            print(f'Auto-correlation values(abs):{acf_values[0, :2]} ~ {acf_values[0, -2:]}')
        else:
            acf_values = np.stack(acf_values, axis=0).mean(axis=0)
            loss = AutoCon(args.batch_size, args.seq_len, np.abs(acf_values), temperature=1.0, base_temperature=1.0)
            print(f'Auto-correlation values(abs):{acf_values[:2]} ~ {acf_values[-2:]}')
        return loss

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        print(f'model parameters:{self.count_parameters(model)}')
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (timeindex, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                if self.args.AutoCon:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs, repr = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_log = dict()
            train_log['loss'] = []
            train_log['MSE_loss'] = []
            train_log['AutoCon_loss'] = []
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            time_now = time.time()
            for i, (timeindex, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                B, T, C = batch_x.shape

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if not self.args.AutoCon:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs, repr = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                MSE_loss = F.mse_loss(outputs, batch_y, reduction='none')

                features = F.normalize(repr, dim=-1)  # B, T, C
                global_pos_labels = timeindex.long()
                local_loss, global_loss = self.AutoCon_loss(features, global_pos_labels)

                if self.args.model == 'AutoConCI':
                    autocon_loss = (local_loss.reshape(B, C, T//3).mean(dim=2).mean(dim=1) + global_loss.mean(dim=0))/2.0
                else:
                    autocon_loss = (local_loss.mean(dim=1) + global_loss)/2.0

                loss = MSE_loss.mean() + self.args.AutoCon_lambda * autocon_loss.mean()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                train_log['loss'].append(loss.item())
                train_log['AutoCon_loss'].append(autocon_loss.detach().cpu())
                train_log['MSE_loss'].append(MSE_loss.detach().cpu())

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_log['loss'] = np.average(train_log['loss'])
            train_log['AutoCon_loss'] = torch.cat(train_log['AutoCon_loss'], dim=0)
            train_log['MSE_loss'] = torch.cat(train_log['MSE_loss'], dim=0)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} "
                  f"Train Loss: {train_log['loss']:.4f} (Forecasting Loss:{train_log['MSE_loss'].mean():.4f} + "
                  f"AutoCon Loss:{train_log['AutoCon_loss'].mean():.4f} x Lambda({self.args.AutoCon_lambda})), "
                  f"Vali MSE Loss: {vali_loss:.4f} Test MSE Loss: {test_loss:.4f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (timeindex, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                timeindex = timeindex.float().to(self.device)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if not self.args.AutoCon:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs, reprs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        # dilate_e, shape_e, temporal_e = shape_metric(preds, trues)  # These metrics take a long time to calculate.
        dilate_e, shape_e,temporal_e  = 0.0,  0.0,  0.0

        print(f'mse:{mse}, mae:{mae}, mape:{mape}, mspe:{mspe} dilate:{dilate_e:.7f}, Shapedtw:{shape_e:.7f}, Temporaldtw:{temporal_e:.7f}')
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        if self.args.save:
            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        return mse, mae, mape, mspe, dilate_e, shape_e, temporal_e
