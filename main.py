from model import UNet
from utility_function import *
from config import Config
import os

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 主函数
def main():
    config = Config()
    # 设置参数
    image_dir = config.image_dir
    mask_dir = config.mask_dir
    # 验证掩码目录是否存在，如果不存在则尝试常见的替代路径
    if not os.path.exists(mask_dir):
        alternative_paths = ['./trimaps', './masks', '../annotations/trimaps']
        for path in alternative_paths:
            if os.path.exists(path):
                mask_dir = path
                print(f"警告：未找到指定的掩码目录，已自动使用替代路径: {mask_dir}")
                break

    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备信息: {device}, CUDA可用: {torch.cuda.is_available()}, 设备数量: {torch.cuda.device_count()}, 可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB')

    # 检查数据目录是否存在
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f'Error: Data directories not found. Please make sure {image_dir} and {mask_dir} exist.')
        return

    # 加载数据集
    print("成功加载数据集！")
    # 使用优化后的load_dataset函数加载数据
    train_loader, test_loader = load_dataset(
        image_dir,
        mask_dir,
        batch_size=config.batch_size,
        img_size=config.img_size,
        test_size=config.test_size
    )

    if not train_loader or not test_loader:
        print("无法加载数据集，程序将退出")
        return

    # 获取数据集大小信息
    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    print(f"数据集数量: {train_size}, 测试集数量: {test_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"训练中每Epoch的batch数: {len(train_loader)}")
    print(f"测试中每Epoch的batch数: {len(test_loader)}")

    if train_loader is None or test_loader is None:
        print("Error: Failed to load dataset. No valid images with masks found.")
        return

    # 创建模型
    print('构建模型...')
    model = UNet(in_channels=3, out_channels=3)

    # 保存模型结构
    model.save_structure()

    # 训练模型
    print('训练模型...')
    model = train_model(model, train_loader, test_loader, config.num_epochs, config.lr, device)

    # 保存模型
    torch.save(model.state_dict(), 'unet_segmentation_model.pth')
    print('最佳模型已保存为 best_model.pth')
    print('最终模型已保存为 unet_segmentation_model.pth')


if __name__ == '__main__':
    main()