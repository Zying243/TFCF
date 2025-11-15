import argparse
import os
import torch
import time
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np
from utils.timer import Timer
from thop import profile, clever_format  # 新增：导入thop工具

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    total_train_timer = Timer()
    total_test_timer = Timer()

    parser = argparse.ArgumentParser(description='FreTS with Sparse Regularization')
    # 基础配置
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='任务名称，可选：long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1, help='训练状态：1-训练，0-测试')
    parser.add_argument('--model_id', type=str, default='Net', help='模型ID')
    parser.add_argument('--model', type=str, default='Net', help='模型名称')
    parser.add_argument('--data', type=str, default='AK_Anchorage', help='dataset type')
    parser.add_argument('--root_path', type=str, default='dataset/Build', help='数据文件根路径')
    parser.add_argument('--data_path', type=str, default='AK_Anchorage.csv', help='数据文件名')
    parser.add_argument('--features', type=str, default='M',
                        help='预测任务类型，可选：M(多变量预测多变量)、S(单变量预测单变量)、MS(多变量预测单变量)')
    parser.add_argument('--target', type=str, default='OT', help='目标特征')
    parser.add_argument('--freq', type=str, default='h',
                        help='时间特征编码频率，可选：s(秒)、t(分钟)、h(小时)、d(天)、b(工作日)、w(周)、m(月)')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型检查点保存路径')
    # 预测任务配置
    parser.add_argument('--seq_len', type=int, default=48, help='输入序列长度')
    parser.add_argument('--label_len', type=int, default=24, help='起始标记长度')
    parser.add_argument('--pred_len', type=int, default=24, help='预测序列长度')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='M4数据集子集')
    parser.add_argument('--inverse', action='store_true', help='是否反转输出数据', default=False)
    # 模型定义
    parser.add_argument('--top_k', type=int, default=5, help='TimesBlock参数')
    parser.add_argument('--num_kernels', type=int, default=6, help='Inception模块核数量')
    parser.add_argument('--enc_in', type=int, default=10, help='编码器输入维度')
    parser.add_argument('--dec_in', type=int, default=1, help='解码器输入维度')
    parser.add_argument('--c_out', type=int, default=1, help='输出维度')
    parser.add_argument('--d_model', type=int, default=128, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数量')
    parser.add_argument('--e_layers', type=int, default=3, help='编码器层数')
    parser.add_argument('--d_layers', type=int, default=2, help='解码器层数')
    parser.add_argument('--d_ff', type=int, default=1280, help='前馈网络维度')
    parser.add_argument('--moving_avg', type=int, default=20, help='移动平均窗口大小')
    parser.add_argument('--factor', type=int, default=1, help='注意力因子')
    parser.add_argument('--distil', action='store_false',
                        help='是否在编码器中使用蒸馏', default=True)
    parser.add_argument('--embed', type=str, default='timeF',
                        help='时间特征编码方式，可选：timeF, fixed, learned')
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数')
    parser.add_argument('--output_attention', action='store_true',
                        help='是否输出编码器注意力权重', default=False)
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='FreTS模型通道独立性：1-独立，0-依赖')
    parser.add_argument('--embed_size', type=int, default=512, help='嵌入层维度')
    parser.add_argument('--hidden_size', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=1, help='Dropout比率')
    parser.add_argument('--sparse_reg', type=float, default=0.1, help='稀疏损失权重（建议0.01-0.5）')
    # 去平稳投影器参数
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='投影器隐藏层维度列表')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='投影器隐藏层层数')
    parser.add_argument('--seg_len', type=int, default=48, help='时间序列分段长度')
    # 优化配置
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载器工作线程数')
    parser.add_argument('--itr', type=int, default=1, help='实验重复次数')
    parser.add_argument('--train_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心值')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='学习率')
    parser.add_argument('--des', type=str, default='test', help='实验描述')
    parser.add_argument('--loss', type=str, default='MSE', help='损失函数，可选：MSE, MAE')
    parser.add_argument('--lradj', type=str, default='type1', help='学习率调整策略')
    parser.add_argument('--use_amp', action='store_true', help='是否使用自动混合精度训练', default=False)
    # GPU配置
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU编号')
    parser.add_argument('--use_multi_gpu', action='store_true', help='是否使用多GPU', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='多GPU设备ID列表')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)
    Exp = Exp_Long_Term_Forecast

    if args.is_training == 1:
        total_train_timer.start()
        for ii in range(args.itr):
            exp = Exp(args)
            setting = '{}_{}_{}_sl{}_pl{}_es{}_hs{}_sr{}_bs{}_lr{}_drop{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.pred_len,
                args.embed_size,
                args.hidden_size,
                args.sparse_reg,
                args.batch_size,
                args.learning_rate,
                args.dropout,
                ii
            )
            print('>>>>>>>开始训练: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

            # ========== 新增：统计模型参数量和FLOPs ==========
            # 1. 构造主输入 x 的虚拟数据（batch_size=1, seq_len=args.seq_len, enc_in=args.enc_in）
            dummy_x = torch.randn(1, args.seq_len, args.enc_in).float()
            # 2. 构造时间标记 x_mark_enc 的虚拟数据（形状：batch_size=1, seq_len=args.seq_len, mark_dim=4）
            # 注：mark_dim（时间标记维度）需与真实数据一致，通常为4（如小时、天、周、月编码），若实际不同可调整
            dummy_x_mark_enc = torch.randn(1, args.seq_len, 4).float()
            # 3. 解码器输入 x_dec、x_mark_dec 按训练逻辑传 None
            dummy_x_dec = None
            dummy_x_mark_dec = None

            if args.use_gpu and torch.cuda.is_available():
                dummy_x = dummy_x.cuda(args.gpu)
                dummy_x_mark_enc = dummy_x_mark_enc.cuda(args.gpu)

            flops, params = profile(
                exp.model,
                inputs=(dummy_x, dummy_x_mark_enc, dummy_x_dec, dummy_x_mark_dec)
            )
            flops, params = clever_format([flops, params], "%.2f")
            print(f"模型参数量: {params}")
            print(f"浮点运算次数: {flops}")


            # ================================================

            def custom_train(exp, setting):
                model = exp.model
                device = exp.device if hasattr(exp, 'device') else torch.device('cpu')
                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
                criterion = torch.nn.MSELoss() if args.loss == 'MSE' else torch.nn.L1Loss()
                train_dataset, train_loader = exp._get_data(flag='train')
                for epoch in range(args.train_epochs):
                    model.train()
                    total_loss = 0.0
                    main_loss = 0.0
                    sparse_loss = 0.0
                    batch_count = 0
                    for batch in train_loader:
                        if len(batch) != 4:
                            raise ValueError("批次数据应包含4个元素：x, y, x_mark, y_mark")
                        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                        batch_x = batch_x.float().to(device)
                        batch_y = batch_y.float().to(device)
                        batch_x_mark = batch_x_mark.float().to(device)
                        outputs = model(batch_x, batch_x_mark, None, None)
                        f_dim = -1 if args.features == 'MS' else 0
                        target = batch_y[:, -args.pred_len:, f_dim:].to(device)
                        main_loss_batch = criterion(outputs, target)
                        if hasattr(model, 'get_sparse_loss'):
                            sparse_loss_batch = model.get_sparse_loss()
                        else:
                            sparse_loss_batch = torch.tensor(0.0, device=device)
                        total_loss_batch = main_loss_batch + args.sparse_reg * sparse_loss_batch
                        optimizer.zero_grad()
                        total_loss_batch.backward()
                        optimizer.step()
                        total_loss += total_loss_batch.item()
                        main_loss += main_loss_batch.item()
                        sparse_loss += sparse_loss_batch.item() if isinstance(sparse_loss_batch,
                                                                              torch.Tensor) else sparse_loss_batch
                        batch_count += 1
                    if (epoch + 1) % 10 == 0:
                        print(f'Epoch [{epoch + 1}/{args.train_epochs}], '
                              f'主损失: {main_loss / batch_count:.4f}, '
                              f'稀疏损失: {sparse_loss / batch_count:.4f}, '
                              f'总损失: {total_loss / batch_count:.4f}')


            custom_train(exp, setting)
            print('>>>>>>>开始测试: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

            total_test_timer.start()
            test_start = time.time()
            exp.test(setting)
            total_test_timer.end(accumulate=True)
            test_elapsed = time.time() - test_start
            print(f"单次测试总耗时: {test_elapsed:.4f} 秒")

            torch.cuda.empty_cache()

        total_train_timer.end()
        print("\n" + "=" * 50)
        total_train_timer.print_stats("总训练")
        total_test_timer.print_stats("总测试")
        print("=" * 50)

    else:
        ii = 0
        setting = '{}_{}_{}_sl{}_pl{}_es{}_hs{}_sr{}_bs{}_lr{}_drop{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.embed_size,
            args.hidden_size,
            args.sparse_reg,
            args.batch_size,
            args.learning_rate,
            args.dropout,
            ii
        )
        exp = Exp(args)
        print('>>>>>>>开始测试: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

        # ========== 新增：测试模式下统计模型参数量和FLOPs ==========
        dummy_input = torch.randn(1, args.seq_len, args.enc_in).float()
        if args.use_gpu and torch.cuda.is_available():
            dummy_input = dummy_input.cuda(args.gpu)
        flops, params = profile(exp.model, inputs=(dummy_input,))
        flops, params = clever_format([flops, params], "%.2f")
        print(f"模型参数量: {params}")
        print(f"浮点运算次数: {flops}")
        # ================================================

        total_test_timer.start()
        exp.test(setting, test=1)
        total_test_timer.end()

        print("\n" + "=" * 50)
        total_test_timer.print_stats("测试")
        print("=" * 50)

        torch.cuda.empty_cache()