import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import json
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap
from config import Config
config = Config()

# 定义Dice损失函数
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets, smooth=1):
        inputs = F.softmax(inputs, dim=1)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        targets = targets.contiguous()
        class_dice = []
        weights = []
        for cls in torch.unique(targets):
            inputs_cls = inputs[..., cls]  # 修复维度访问错误
            targets_cls = (targets == cls).float()
            intersection = (inputs_cls * targets_cls).sum()
            dice = (2. * intersection + smooth) / (inputs_cls.sum() + targets_cls.sum() + smooth)
            class_dice.append(dice)
            if self.weight is not None:
                weights.append(self.weight[cls])

        if self.weight is not None and len(weights) > 0:
            weights = torch.tensor(weights, device=inputs.device)
            # 使用原始权重比例而非归一化
            return 1 - torch.mean(torch.stack(class_dice) * weights)
        else:
            return 1 - torch.mean(torch.stack(class_dice)) if class_dice else torch.tensor(1.0)  # 无类别时返回1.0使损失为0

# 自定义数据集类，用于加载本地图像和掩码
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_list, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = image_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        # 正确处理各种图像扩展名，生成对应的掩码文件名
        base_name = os.path.splitext(self.images[idx])[0]
        mask_filename = f"{base_name}.png"
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # 加载图像和掩码
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 转换为灰度图像

        # 处理Oxford-IIIT Pet Dataset的掩码值（1:前景, 2:背景, 3:边界）
        mask = np.array(mask)
        # 按照用户定义转换掩码值：
        # 1=前景(Foreground) → 0
        # 2=背景(Background) → 1
        # 3=未分类(Not classified) → 2
        mask = np.where(mask == 3, 2, mask - 1)

        # 验证掩码值是否有效
        if not np.all(np.isin(mask, [0, 1, 2])):
            invalid_values = np.unique(mask[~np.isin(mask, [0, 1, 2])])
            print(f"警告: 掩码中存在无效值 {invalid_values}，样本索引: {idx}")

        mask = Image.fromarray(mask.astype(np.uint8))

        # 应用变换
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        mask = mask.squeeze(0)  # 移除通道维度

        return image, mask

# 图像预处理和数据增强
def get_transforms(img_size=256):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # 扩大裁剪范围增加多样性
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),  # 增加旋转角度范围
        transforms.RandomAffine(degrees=0, shear=10),  # 添加剪切变换
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # 添加透视变换
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # 增强颜色抖动
        transforms.RandomGrayscale(p=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33)),  # 增加随机擦除概率和范围
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size))
    ])

    return train_transform, test_transform, mask_transform

# 加载数据集
def load_dataset(image_dir, mask_dir, batch_size=8, img_size=256, test_size=0.2):
    # 获取所有图像文件名并检查对应的掩码文件是否存在
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # 用户已明确指定trimaps目录，无需额外搜索子目录
    # 验证掩码目录是否存在
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"掩码目录不存在: {mask_dir}")

    # 直接在指定的trimaps目录中检查掩码文件
    all_images = []
    missing_masks = []
    for f in image_files:
        base_name = os.path.splitext(f)[0]
        found = False
        for ext in ['.png', '.jpg', '.jpeg']:
            mask_filename = f"{base_name}{ext}"
            mask_path = os.path.join(mask_dir, mask_filename)
            if os.path.exists(mask_path):
                all_images.append(f)
                found = True
                break
        if not found:
            missing_masks.append(f)

    # 警告缺失的掩码文件
    if missing_masks:
        print(f"警告: {len(missing_masks)}个图像缺少对应的掩码文件")
        print(f"示例缺失文件: {missing_masks[:5]}" if len(missing_masks) > 5 else f"缺失文件: {missing_masks}")

    if not all_images:
        raise ValueError("未找到任何包含对应掩码的图像文件")

    # 分割训练集和测试集
    train_images, test_images = train_test_split(all_images, test_size=test_size, random_state=42)

    # 获取变换
    train_transform, test_transform, mask_transform = get_transforms(img_size)

    # 创建数据集和数据加载器 - 移除临时目录和符号链接逻辑
    train_dataset = SegmentationDataset(
        image_dir,
        mask_dir,
        train_images,  # 传递训练图像列表
        transform=train_transform,
        mask_transform=mask_transform
    )

    test_dataset = SegmentationDataset(
        image_dir,
        mask_dir,
        test_images,  # 传递测试图像列表
        transform=test_transform,
        mask_transform=mask_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

# 加载测试数据集
def load_test_dataset(image_dir, mask_dir, img_size=256):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    all_images = []

    for f in image_files:
        base_name = os.path.splitext(f)[0]
        found = False
        for ext in ['.png', '.jpg', '.jpeg']:
            mask_filename = f"{base_name}{ext}"
            if os.path.exists(os.path.join(mask_dir, mask_filename)):
                all_images.append(f)
                found = True
                break

    _, test_transform, mask_transform = get_transforms(img_size)
    dataset = SegmentationDataset(image_dir, mask_dir, all_images, test_transform, mask_transform)
    return DataLoader(dataset, batch_size=8, shuffle=False)

# 可视化结果
def visualize_results(model, test_loader, num_samples=5, device='cuda'):
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(test_loader))
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        outputs = torch.argmax(outputs, dim=1)

        images = images.cpu().numpy()
        masks = masks.cpu().numpy()
        outputs = outputs.cpu().numpy()

        plt.figure(figsize=(15, 10))
        for i in range(min(num_samples, len(images))):
            # 反归一化
            img = images[i].transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            plt.subplot(num_samples, 3, i * 3 + 1)
            plt.imshow(img)
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(num_samples, 3, i * 3 + 2)
            cmap = ListedColormap(['#000000', '#646464', '#FFFFFF'])  # 黄色:前景, 蓝色:背景, 红色:未分类
            plt.imshow(masks[i], cmap=cmap)
            cbar = plt.colorbar(ticks=[0, 1, 2], label='Class')
            cbar.ax.set_yticklabels(['Foreground', 'Background', 'Edge'],
                                    fontproperties=FontProperties(family='Microsoft YaHei', size=8))
            plt.title('Ground Truth')
            plt.axis('off')

            plt.subplot(num_samples, 3, i * 3 + 3)
            plt.imshow(outputs[i], cmap=cmap)
            cbar = plt.colorbar(ticks=[0, 1, 2], label='Class')
            cbar.ax.set_yticklabels(['Foreground', 'Background', 'Edge'],
                                    fontproperties=FontProperties(family='Microsoft YaHei', size=8))
            plt.title('Prediction')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('segmentation_visualization.png')
        plt.close()
        print(f"可视化结果已保存至 segmentation_visualization.png")

# 计算类别权重以解决类别不平衡问题
def compute_class_weights(dataset, num_classes=3):
    class_counts = torch.zeros(num_classes)
    for _, mask in tqdm(dataset, desc="计算类别权重", total=len(dataset), ncols=150):
        mask = mask.flatten()
        for cls in range(num_classes):
            class_counts[cls] += (mask == cls).sum()
    total = class_counts.sum()
    weights = total / (num_classes * class_counts + 1e-6)
    return weights

# 计算mIoU指标
def compute_miou(preds, targets, num_classes=3):
    ious = []
    preds = F.softmax(preds, dim=1).argmax(dim=1)

    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union == 0:
            ious.append(torch.tensor(1.0) if intersection == 0 else torch.tensor(0.0))
        else:
            ious.append((intersection + 1e-6) / (union + 1e-6))

    return torch.mean(torch.stack(ious)) if ious else torch.tensor(0.)

# 训练模型
def train_model(model, train_loader, test_loader, num_epochs=50, lr=0.001, device='cuda', patience=config.patience, min_delta=config.min_delta,):
    # 计算类别权重
    WEIGHTS_FILE = "class_weights.json"

    # 检查是否存在保存的类别权重文件
    if os.path.exists(WEIGHTS_FILE):
        print(f"加载本地类别权重文件: {WEIGHTS_FILE}")
        with open(WEIGHTS_FILE, 'r') as f:
            class_weights = torch.tensor(json.load(f))
    else:
        # 计算类别权重
        print("计算类别权重以解决类别不平衡问题...")
        class_weights = compute_class_weights(train_loader.dataset)
        # 保存类别权重到本地文件
        with open(WEIGHTS_FILE, 'w') as f:
            json.dump(class_weights.tolist(), f)
        print(f"类别权重已保存到 {WEIGHTS_FILE}")
    print(f"计算得到的类别权重: {class_weights.tolist()}")

    # 定义损失函数和优化器
    # 添加标签平滑以减少过拟合
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    criterion_dice = DiceLoss(weight=class_weights.to(device))
    def combined_loss(inputs, targets, alpha=0.5):
        ce_loss = criterion_ce(inputs, targets)
        dice_loss = criterion_dice(inputs, targets)
        # 加权组合损失函数，alpha控制交叉熵损失权重
        return alpha * ce_loss + (1 - alpha) * dice_loss
    criterion = combined_loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    # 使用余弦退火调度器替代ReduceLROnPlateau，提供更灵活的学习率调整
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # 将模型移至指定设备
    model.to(device)

    # 早停机制初始化
    best_test_loss = float('inf')
    early_stop_counter = 0
    best_model_path = 'best_model.pth'

    # 记录训练和验证指标
    train_losses = []
    test_losses = []
    test_mious = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        # 创建进度条对象，添加动态loss显示，并设置较长的进度条宽度
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} 训练", total=len(train_loader), ncols=150)

        for batch_idx, (images, masks) in enumerate(train_pbar):
            # 将数据移至指定设备
            images, masks = images.to(device), masks.to(device)

            # 前向传播
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            # 反向传播和优化
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # 梯度裁剪以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

            # 在进度条中动态显示当前batch的loss值
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # 测试阶段
        model.eval()
        test_loss = 0.0
        total_miou = 0.0

        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} 测试", total=len(test_loader), ncols=150):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                test_loss += loss.item() * images.size(0)

                # 计算mIoU
                miou = compute_miou(outputs, masks)
                total_miou += miou.item() * images.size(0)

        # 计算平均测试损失
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        # 计算平均mIoU
        avg_miou = total_miou / len(test_loader.dataset)
        test_mious.append(avg_miou)

        # 打印训练进度
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test mIoU: {avg_miou:.4f}')
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(test_loss)
        current_lr = scheduler.get_last_lr()[0]
        if current_lr != prev_lr:
            print(f"学习率已调整为: {current_lr}")

        # 早停检查
        if test_loss < best_test_loss - min_delta:
            best_test_loss = test_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型至 {best_model_path}，验证损失: {best_test_loss:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"早停计数 {early_stop_counter}/{patience} ")
            if early_stop_counter >= patience:
                print(f"早停触发！在第 {epoch+1} 轮停止训练")
                break

        # 每5个epoch打印学习率
        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr}")

    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))

    # 绘制训练指标曲线
    plt.figure(figsize=(15, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(test_losses)+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()

    # mIoU曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(test_mious)+1), test_mious, label='Test mIoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('Test Mean Intersection over Union')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

    return model