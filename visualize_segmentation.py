import matplotlib
import argparse
from model import UNet
from utility_function import *

matplotlib.use('Agg')

# 调整字体顺序，优先使用Windows系统常见中文字体
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def main():
    parser = argparse.ArgumentParser(description='可视化图像分割结果')
    parser.add_argument('--model-path', type=str, default='best_model.pth', help='模型权重路径')
    parser.add_argument('--image-dir', type=str, default='./images', help='图像目录')
    parser.add_argument('--mask-dir', type=str, default='./annotations/trimaps', help='掩码目录')
    parser.add_argument('--img-size', type=int, default=256, help='图像大小')
    parser.add_argument('--num-samples', type=int, default=5, help='可视化样本数')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载模型
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f'已加载模型: {args.model_path}')
    
    # 加载测试数据
    test_loader = load_test_dataset(args.image_dir, args.mask_dir, args.img_size)
    print(f'测试数据集大小: {len(test_loader.dataset)}')
    
    # 可视化结果
    visualize_results(model, test_loader, args.num_samples, device)

if __name__ == '__main__':
    main()