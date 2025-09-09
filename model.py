import torch.nn as nn
import os
from contextlib import redirect_stdout

# 注意力门控机制
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# 带注意力机制的U-Net模型
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        # 下采样路径（编码器）
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # 瓶颈
        self.bottleneck = self.conv_block(512, 1024)

        # 注意力门控
        self.attention_gate4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.attention_gate3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.attention_gate2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.attention_gate1 = AttentionGate(F_g=64, F_l=64, F_int=32)

        # 上采样路径（解码器）
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        # 输出层
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)  # 根据参数动态设置输出通道数

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # 添加一个方法来保存模型结构，而不是在构造函数中直接保存
    def save_structure(self, file_path='model_structure.txt'):
        try:
            with open(file_path, 'w') as f:
                with redirect_stdout(f):
                    print(self)
            print(f"模型结构已保存到 {os.path.abspath(file_path)}")
            return True
        except Exception as e:
            print(f"保存模型结构时出错: {e}")
            return False

    def conv_block(self, in_channels, out_channels, use_residual=True):
        # 主路径
        main_path = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        ]

        main_path = nn.Sequential(*main_path)

        # 残差连接路径
        if use_residual:
            if in_channels != out_channels:
                # 如果通道数不同，使用1x1卷积调整通道数
                shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                # 如果通道数相同，直接使用恒等映射
                shortcut = nn.Identity()

            # 创建带残差连接的自定义模块
            return ResidualBlock(main_path, shortcut)

        # 如果不使用残差连接，添加ReLU并返回主路径
        return nn.Sequential(main_path, nn.ReLU(inplace=True))

    def forward(self, x):
        # 下采样
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        # 瓶颈
        bottleneck = self.bottleneck(self.pool(enc4))

        # 上采样并使用注意力门控
        dec4 = self.upconv4(bottleneck)
        # 添加注意力门控
        att4 = self.attention_gate4(dec4, enc4)
        dec4 = torch.cat((dec4, att4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        att3 = self.attention_gate3(dec3, enc3)
        dec3 = torch.cat((dec3, att3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        att2 = self.attention_gate2(dec2, enc2)
        dec2 = torch.cat((dec2, att2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        att1 = self.attention_gate1(dec1, enc1)
        dec1 = torch.cat((dec1, att1), dim=1)
        dec1 = self.decoder1(dec1)

        # 输出
        out = self.outconv(dec1)
        return out

# 自定义残差块模块
class ResidualBlock(nn.Module):
    def __init__(self, main_path, shortcut):
        super(ResidualBlock, self).__init__()
        self.main_path = main_path
        self.shortcut = shortcut
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.main_path(x)
        out += residual  # 残差连接
        out = self.relu(out)
        return out


# 测试前向传播是否正常
if __name__ == "__main__":
    import torch

    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)

    # 创建测试输入 (batch_size=1, channels=3, height=256, width=256)
    test_input = torch.randn(1, 3, 256, 256)

    # 创建模型实例
    model = UNet(in_channels=3, out_channels=3)

    # 移动到可用设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_input = test_input.to(device)

    try:
        # 执行前向传播
        with torch.no_grad():
            output = model(test_input)

        # 检查输出形状
        assert output.shape == (1, 3, 256, 256), f"前向传播输出形状错误: {output.shape}，期望 (1, 3, 256, 256)"

        # 检查输出是否包含NaN或无穷大值
        assert not torch.isnan(output).any(), "前向传播输出包含NaN值"
        assert not torch.isinf(output).any(), "前向传播输出包含无穷大值"

        print(f"前向传播测试成功! 输入形状: {test_input.shape}, 输出形状: {output.shape}")
        print(f"使用设备: {device}")
    except Exception as e:
        print(f"前向传播测试失败: {str(e)}")
