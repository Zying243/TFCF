import os
import torch
from models.TexFilter import Model as TexFilter
from models.DLinear import Model as DLinear
from models.FreTS import Model as FreTS
from models.PaiFilter import Model as PaiFilter

class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'DLinear': DLinear,
            'FreTS': FreTS,
            'TexFilter': TexFilter,
            'PaiFilter': PaiFilter
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        # 检查模型是否成功移至目标设备
        self._verify_model_device()

    def _build_model(self):
        try:
            model_type = self.args.model
            if model_type not in self.model_dict:
                raise ValueError(
                    f"Model type {model_type} not supported. Available models: {list(self.model_dict.keys())}")

            # 假设所有模型类都接受config参数
            model = self.model_dict[model_type](self.args)
            return model
        except Exception as e:
            print(f"Error building model: {e}")
            raise

    def _acquire_device(self):
        if self.args.use_gpu:
            # 检查CUDA是否可用
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, using CPU instead.")
                return torch.device('cpu')

            if not self.args.use_multi_gpu:
                # 单GPU设置
                gpu_id = int(self.args.gpu)
                if gpu_id >= torch.cuda.device_count():
                    raise ValueError(f"GPU ID {gpu_id} not available. Total GPUs: {torch.cuda.device_count()}")
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                device = torch.device(f'cuda:{gpu_id}')
                print(f'Use GPU: cuda:{gpu_id} ({torch.cuda.get_device_name(gpu_id)})')
            else:
                # 多GPU设置
                device_ids = [int(id) for id in self.args.devices.split(',')]
                available_gpus = torch.cuda.device_count()
                invalid_ids = [id for id in device_ids if id >= available_gpus]
                if invalid_ids:
                    raise ValueError(f"Invalid GPU IDs: {invalid_ids}. Available GPUs: {list(range(available_gpus))}")
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                device = torch.device('cuda:0')  # 默认使用第一个可见GPU
                print(f'Use Multi-GPU: {device_ids}')
                print(f'Primary GPU: cuda:0 ({torch.cuda.get_device_name(0)})')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _verify_model_device(self):
        # 验证模型参数是否在正确的设备上
        first_param_device = next(self.model.parameters()).device
        if first_param_device != self.device:
            print(f"Warning: Model not on expected device. Expected: {self.device}, Actual: {first_param_device}")

    def _get_data(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def vali(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def train(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def test(self):
        raise NotImplementedError("Subclasses should implement this method.")