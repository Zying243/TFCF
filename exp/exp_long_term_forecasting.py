import os
import time
import warnings
import logging
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.timer import Timer  # 确保导入Timer类

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')


def setup_logger(setting):
    log_dir = os.path.join('./logs', setting)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        args = self._init_args(args)
        super().__init__(args)
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
        self.logger = setup_logger(args.model + '_' + args.data)
        self.logger.info(f'Args: {args}')

        # 类内全局计时器（无需外部传递）
        self.total_train_timer = Timer()
        self.epoch_timer = Timer()
        self.total_test_timer = Timer()
        self.infer_timer = Timer()  # 推理计时器，内部使用

    def _init_args(self, args):
        if not hasattr(args, 'pin_memory'):
            args.pin_memory = True
        if not hasattr(args, 'num_workers'):
            args.num_workers = 4
        if not hasattr(args, 'weight_decay'):
            args.weight_decay = 0.0
        if not hasattr(args, 'lr_scheduler'):
            args.lr_scheduler = 'plateau'
        if not hasattr(args, 'grad_clip'):
            args.grad_clip = None
        if not hasattr(args, 'save_freq'):
            args.save_freq = 10
        if not hasattr(args, 'use_amp'):
            args.use_amp = False
        if not hasattr(args, 'progress_bar'):
            args.progress_bar = True
        return args

    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(self, 'logger'):
            self.logger.info("资源清理完成")
        # 重置计时器
        self.total_train_timer.reset()
        self.epoch_timer.reset()
        self.total_test_timer.reset()
        self.infer_timer.reset()

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        if flag == 'train' and hasattr(data_loader, 'shuffle'):
            data_loader.shuffle = True
        return data_set, data_loader

    def _select_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.args.learning_rate,
                               weight_decay=self.args.weight_decay)
        if self.args.lr_scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=self.args.patience // 2, factor=0.5)
        elif self.args.lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args.train_epochs, eta_min=1e-6)
        else:
            scheduler = None
        return optimizer, scheduler

    def _select_criterion(self):
        if self.args.loss == 'mse':
            criterion = nn.MSELoss()
        elif self.args.loss == 'mae':
            criterion = nn.L1Loss()
        elif self.args.loss == 'huber':
            criterion = nn.HuberLoss()
        else:
            raise NotImplementedError(f'Loss {self.args.loss} not implemented')
        return criterion

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # 使用类内的infer_timer，无需外部传递
        self.infer_timer.start()
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        self.infer_timer.end(accumulate=True)

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        return outputs, batch_y

    def _model_forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        if self.args.output_attention:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        self.infer_timer.reset()  # 重置推理计时器
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                outputs, batch_y = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss.item())
        # 打印验证阶段推理统计
        avg_infer_time = self.infer_timer.get_avg_time()
        self.logger.info(f"验证阶段 - 累计批次: {self.infer_timer.count}, 平均推理耗时: {avg_infer_time:.6f} 秒")

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        self.total_train_timer.start()
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        try:
            model_optim, scheduler = self._select_optimizer()
            criterion = self._select_criterion()
            early_stopping = EarlyStopping(patience=self.args.patience,
                                           verbose=True,
                                           delta=1e-4,
                                           restore_best_weights=True)

            para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
            self.logger.info(f'Model {self.args.model} params: {(para * 4 / 1024 / 1024):.4f}M')

            start_epoch = 0
            best_val_loss = float('inf')
            checkpoint_path = os.path.join(path, 'latest_checkpoint.pth')

            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scaler and 'scaler_state_dict' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_epoch = checkpoint['epoch']
                best_val_loss = checkpoint['best_val_loss']
                self.logger.info(f"Resuming training from epoch {start_epoch}")

            train_steps = len(train_loader)
            self.model.train()

            for epoch in range(start_epoch, self.args.train_epochs):
                self.epoch_timer.start()
                iter_count = 0
                train_loss = []
                self.infer_timer.reset()  # 重置单epoch推理计时器

                if self.args.progress_bar:
                    train_loader = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.args.train_epochs}')

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()

                    outputs, batch_y = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                    if (i + 1) % self.args.accumulation_steps == 0:
                        if self.args.use_amp:
                            self.scaler.scale(loss).backward()
                            self.scaler.unscale_(model_optim)
                            if self.args.grad_clip:
                                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                            self.scaler.step(model_optim)
                            self.scaler.update()
                        else:
                            loss.backward()
                            if self.args.grad_clip:
                                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                            model_optim.step()

                    if (i + 1) % 100 == 0:
                        self.logger.info(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")

                # 打印epoch统计
                epoch_elapsed = self.epoch_timer.end(accumulate=False)
                avg_infer_time = self.infer_timer.get_avg_time()
                self.logger.info(
                    f"Epoch: {epoch + 1} - 单轮耗时: {epoch_elapsed:.2f}s, 训练推理平均耗时: {avg_infer_time:.6f}s")

                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                self.logger.info(
                    f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

                if scheduler:
                    if self.args.lr_scheduler == 'plateau':
                        scheduler.step(vali_loss)
                    else:
                        scheduler.step()
                else:
                    adjust_learning_rate(model_optim, epoch + 1, self.args)

                if (epoch + 1) % self.args.save_freq == 0:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': model_optim.state_dict(),
                        'best_val_loss': best_val_loss,
                        'scaler_state_dict': self.scaler.state_dict() if self.args.use_amp else None,
                    }, os.path.join(path, f'checkpoint_epoch_{epoch + 1}.pth'))

                early_stopping(vali_loss, self.model, path)
                if vali_loss < best_val_loss:
                    best_val_loss = vali_loss
                    torch.save(self.model.state_dict(), os.path.join(path, 'best_model.pth'))
                    self.logger.info(f'Best model saved at epoch {epoch + 1} with val loss: {best_val_loss:.7f}')

                if early_stopping.early_stop:
                    self.logger.info("Early stopping")
                    break

            # 打印训练总统计
            total_train_elapsed = self.total_train_timer.end()
            self.logger.info("\n" + "=" * 60)
            self.logger.info(f"训练总统计 - 总耗时: {total_train_elapsed:.2f}s, 总epoch数: {epoch + 1 - start_epoch}")
            self.logger.info("=" * 60)

            best_model_path = os.path.join(path, 'best_model.pth')
            self.model.load_state_dict(torch.load(best_model_path))
            return self.model

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            self.cleanup()

    # 核心修复：test方法参数移除infer_timer，直接用self.infer_timer
    def test(self, setting, test=0, save_prediction=True):
        try:
            self.total_test_timer.start()
            self.infer_timer.reset()  # 重置推理计时器（避免残留数据）

            test_data, test_loader = self._get_data(flag='test')
            if test:
                self.logger.info('loading model')
                self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'best_model.pth')))

            preds = []
            trues = []
            folder_path = './test_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    # 调用_process_one_batch，内部已用self.infer_timer统计推理时间
                    outputs, batch_y = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                    preds.append(outputs)
                    trues.append(batch_y)

                    if i % 20 == 0:
                        input = batch_x.detach().cpu().numpy()
                        if test_data.scale and self.args.inverse:
                            shape = input.shape
                            input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                        gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

            # 打印测试总统计
            total_test_elapsed = self.total_test_timer.end()
            avg_infer_time = self.infer_timer.get_avg_time()
            self.logger.info("\n" + "=" * 60)
            self.logger.info(f"测试总统计 - 总耗时: {total_test_elapsed:.2f}s, 累计测试批次: {self.infer_timer.count}")
            self.logger.info(
                f"推理统计 - 平均单批次耗时: {avg_infer_time:.6f}s, 推理总耗时: {self.infer_timer.total_time:.4f}s")
            self.logger.info("=" * 60)

            preds = np.array(preds)
            trues = np.array(trues)
            self.logger.info(f'test shape: {preds.shape} {trues.shape}')
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            self.logger.info(f'reshaped test shape: {preds.shape} {trues.shape}')

            if save_prediction:
                folder_path = './results/' + setting + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                mae, mse, rmse, mape, mspe = metric(preds, trues)
                self.logger.info(f'Metrics: mse:{mse}, mae:{mae}, rmse:{rmse}, mape:{mape}, mspe:{mspe}')
                with open("result_long_term_forecast.txt", 'a') as f:
                    f.write(setting + "  \n")
                    f.write(f'Metrics: mse:{mse}, mae:{mae}, rmse:{rmse}, mape:{mape}, mspe:{mspe}')
                    f.write('\n\n')
                np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
                np.save(folder_path + 'pred.npy', preds)
                np.save(folder_path + 'true.npy', trues)

            return preds, trues

        except Exception as e:
            self.logger.error(f"Testing failed: {str(e)}")
            raise
        finally:
            self.cleanup()