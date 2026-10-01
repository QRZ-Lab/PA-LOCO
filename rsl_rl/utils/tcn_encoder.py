import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class TcnEncoder(nn.Module):
    def __init__(self, num_inputs, history_length, num_outputs, num_channels, kernel_size, dropout=0.2):
        super(TcnEncoder, self).__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1] * history_length, num_outputs)  # Adjust the input size for the linear layer

    def forward(self, x):
        # x shape: (batch_size, num_obs， history_length)
        x = self.tcn(x)
        # Reshape x before passing it to the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        :param num_inputs: int,  输入通道数或者特征数
        :param num_channels: list, 每层的 hidden_channel 数. 例如 [5,12,3], 代表有 3 个 block,
                block1 的输出 channel 数量为 5;
                block2 的输出 channel 数量为 12;
                block3 的输出 channel 数量为 3.
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        构成 TCN 的核心 Block, 原作者在图中成为 Residual block, 是因为它存在残差连接.
        但注意, 这个模块包含了 2 个 Conv1d.

        :param n_inputs: int, 输入通道数或者特征数
        :param n_outputs: int, 输出通道数或者特征数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长, 在TCN固定为1
        :param dilation: int, 膨胀系数. 与这个 Residual block(或者说, 隐藏层)所在的层数有关系.
                例如, 如果这个 Residual block 在第 1 层, dilation = 2**0 = 1;
                     如果这个 Residual block 在第 2 层, dilation = 2**1 = 2;
                     如果这个 Residual block 在第 3 层, dilation = 2**2 = 4;
                     如果这个 Residual block 在第 4 层, dilation = 2**3 = 8 ......
        :param padding: int, 填充系数. 与 kernel_size 和 dilation 有关.
        :param dropout: float, dropout 比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        # 因为 padding 的时候, 在序列的左边和右边都有填充, 所以要裁剪
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 1×1的卷积. 只有在进入Residual block的通道数与出Residual block的通道数不一样时使用.
        # 一般都会不一样, 除非num_channels这个里面的数, 与num_inputs相等. 例如[5,5,5], 并且num_inputs也是5
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        # 在整个Residual block中有非线性的激活. 这个容易忽略!
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# 设置网络参数
num_envs = 16
num_obs = 64
history_length = 50
num_latency = 16
num_channels = [32, 32, 32]  # 你可以根据需要调整通道数
kernel_size = 3
dropout = 0.2

# 创建网络
# 输入 num_obs 个通道/特征
# 构建 1 层的 TCN，最后输出一个通道，或者特征
encoder_layers = [
    TcnEncoder(num_inputs=num_obs, history_length=history_length, num_outputs=num_latency,
               num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
]
encoder = nn.Sequential(*encoder_layers)

# 检测输出
with torch.no_grad():
    # 模型输入一定是 (batch_size, channels, length)
    encoder.eval()
    # print(encoder(torch.randn(num_envs, num_obs, history_length)).shape)
