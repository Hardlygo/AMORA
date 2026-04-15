import datetime
from pathlib import Path
from contextlib import contextmanager

import torch
import torch.nn as nn
import math
from typing import List
from typing import Union
from typing import Optional


def get_run_name(*arguments, **keyword_arguments) -> str:
    name = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    if arguments:
        for argument in arguments:
            name += f'_{str(argument)}'
    if keyword_arguments:
        for argument_name, argument_value in keyword_arguments.items():
            name += f'_{argument_name}{str(argument_value)}'
    return name


@contextmanager
def eval_mode(net: nn.Module):
    """Temporarily switch to evaluation mode."""
    originally_training = net.training
    try:
        net.eval()
        yield net
    finally:
        if originally_training:
            net.train()


def get_device(overwrite: Optional[bool] = None, device: str = 'cuda:0', fallback_device: str = 'cpu') -> torch.device:
    use_cuda = torch.cuda.is_available() if overwrite is None else overwrite
    return torch.device(device if use_cuda else fallback_device)


def update_network_parameters(source: nn.Module, target: nn.Module, tau: float) -> None:
    for target_parameters, source_parameters in zip(target.parameters(), source.parameters()):
        target_parameters.data.copy_(
            target_parameters.data * (1.0 - tau) + source_parameters.data * tau)


def save_model(model: nn.Module, file: Union[str, Path]) -> None:
    try:
        torch.save(model.state_dict(), file)
    except FileNotFoundError:
        print('Unable to save the models')


def load_model(model: nn.Module, file: Union[str, Path]) -> None:
    try:
        model.load_state_dict(torch.load(file))
    except FileNotFoundError:
        print('Unable to load the models')


def save_to_writer(writer, tag_to_scalar_value: dict, step: int) -> None:
    for tag, scalar_value in tag_to_scalar_value.items():
        writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=step)


def filter_info(dictionary: dict) -> dict:
    filtered_dictionary = {}
    for key, value in dictionary.items():
        if '/' in key and isinstance(value, (int, float)):
            filtered_dictionary[key] = value
    return filtered_dictionary


def weight_initialization(module) -> None:
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight, gain=1)
        torch.nn.init.constant_(module.bias, 0)


def get_multilayer_perceptron(unit_list: List[int], keep_last_relu: bool = False) -> nn.Sequential:
    module_list = []
    for in_features, out_features in zip(unit_list, unit_list[1:]):
        module_list.append(nn.Linear(in_features, out_features))
        module_list.append(nn.ReLU())
    if keep_last_relu:
        return nn.Sequential(*module_list)
    else:
        return nn.Sequential(*module_list[:-1])


def get_timedelta_formatted(td):
    hours, rem = divmod(td.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe是二维 max_len*d_model(最大长度和ebed长度)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(
            0, max_len, dtype=torch.float).unsqueeze(1)  # max_len*1
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) * (-math.log(10000.0) / d_model))  # max_len/2
        pe[:, 0::2] = torch.sin(position * div_term)  # 对应位置相乘 ，偶数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数
        pe = pe.unsqueeze(0).transpose(0, 1)  # max_len*1*d_model
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]  # 广播机制 (t,bs,dmodel)
        return self.dropout(x)


def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}!")
