# 定义配置类
class Config:
    def __init__(self):
        # 训练配置
        self.batch_size = 16    # 批次数量
        self.img_size   = 256   # 输入图像尺寸
        self.num_epochs = 100   # 训练步数
        self.lr         = 0.001 # 初始学习率
        self.test_size  = 0.2   # 测试集比例
        self.patience   = 5     # 早停计数限制
        self.min_delta  = 0.001 # 最小下降

        # 数据集位置
        self.image_dir  = './images'                # 修改为用户提供的图像目录
        self.mask_dir   = './annotations/trimaps'   # 修改为用户提供的标注目录（trimaps子目录包含分割掩码）