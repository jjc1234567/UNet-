import torch
from torch import nn
from torchvision.transforms import functional as TVF

# 定义 UNet 模型的基本块
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.2) -> None:
        super().__init__()
        # 两个卷积层和 LeakyReLU 激活函数组成的基本块
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope, True)

    def forward(self, x):
        # 前向传播函数，对输入进行两次卷积和 LeakyReLU 激活
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        return x

# 定义 UNet 模型的上采样块
class UNetUpsample(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        # 转置卷积层用于上采样
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, x1, x2):
        # 前向传播函数，对输入进行上采样并进行特征融合
        x1 = self.up(x1)
        x2 = TVF.center_crop(x2, x1.shape[-2:])
        x = torch.cat([x1, x2], dim=1)
        return x

# 定义整个 UNet 模型
class UNet(nn.Module):
    def __init__(self, depth=5, base_channels=32, in_channels=3, out_channels=3) -> None:
        super().__init__()
        self.depth = depth
        self.pool = nn.MaxPool2d(2)
        x = in_channels
        y = base_channels
        # 定义下采样部分，包括多个 UNetBlock
        for i in range(1, depth + 1):
            self.add_module(f"block_d{i}", UNetBlock(x, y))
            x, y = y, y * 2
        y = x // 2
        # 定义上采样部分，包括多个 UNetBlock 和 UNetUpsample
        for i in range(1, depth):
            self.add_module(f"block_u{i}", UNetBlock(x, y))
            self.add_module(f"upsample{i}", UNetUpsample(x, y))
            x, y = y, y // 2
        # 输出层，1x1 卷积用于减少通道数到输出通道数
        self.head = nn.Conv2d(x, out_channels, 1)

    def forward(self, x: torch.Tensor):
        feat = []
        # 下采样阶段，保存每个阶段的特征图
        for i in range(1, self.depth):
            tmp = self.get_submodule(f"block_d{i}")(x)
            feat.append(tmp)
            x = self.pool(tmp)
        x = self.get_submodule(f"block_d{self.depth}")(x)

        # 上采样阶段，进行特征融合和上采样
        for i in range(1, self.depth):
            x = self.get_submodule(f"block_u{i}")(
                self.get_submodule(f"upsample{i}")(x, feat[-i])
            )
        x = self.head(x)
        return x

# 模型测试
if __name__ == "__main__":
    from torchinfo import summary
    print(torch.__version__)
    # 创建 UNet 模型实例并打印模型信息
    model = UNet(5, 64)
    x = torch.rand((1, 3, 512, 512))
    summary(model, input_data=x)
