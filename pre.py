import os
import torch
from torchvision import transforms
from PIL import Image
from basic.unet import UNet  # 确保这个模块可以被正确导入

def optimize_images(input_folder, model_path, output_folder='./output_results', image_size=(300, 300)):
    """
    优化指定文件夹中的图像，使用给定的模型进行推理，然后将优化后的图像保存到输出文件夹中。

    Parameters:
    - input_folder (str): 输入图像文件夹的路径。
    - model_path (str): 训练好的模型的路径。
    - output_folder (str): 优化后的图像保存的文件夹路径（默认为 './output_results'）。
    - image_size (tuple): 输入图像的尺寸（默认为 (300, 300)）。
    """
    
    # 定义测试数据集类
    class TestDataset:
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.files = os.listdir(root_dir)
            self.transform = transform

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            img_path = os.path.join(self.root_dir, self.files[idx])
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                transformed_image = self.transform(image)
            else:
                transformed_image = image  # 如果没有变换，直接返回原图

            return {'image': transformed_image, 'filename': self.files[idx]}

    # 数据预处理和转换
    test_transform = transforms.Compose([
        transforms.Resize(image_size),  # 与训练时相同的预处理
        transforms.ToTensor(),
    ])

    # 创建测试数据集实例
    test_dataset = TestDataset(root_dir=input_folder, transform=test_transform)

    # 创建测试数据加载器
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 初始化模型
    model = UNet(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 将模型和数据移动到 GPU 上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 对测试数据进行推理
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)

            # 前向传播
            outputs = model(images)
            outputs = torch.nn.functional.interpolate(outputs, size=(images.size(2), images.size(3)), mode='bilinear', align_corners=False)

            # 保存优化后的图片
            filename = batch['filename'][0]
            output_path = os.path.join(output_folder, f"optimized_{filename}")
            output_image = transforms.ToPILImage()(outputs.squeeze(0).cpu())  # 将张量移回 CPU
            output_image.save(output_path)

    print(f"测试完成，优化后的图片已保存在 '{output_folder}' 文件夹中。")

# 示例调用
# optimize_images(input_folder='original_test_images', model_path='larger_unet_lsp_epoch_100.pth')
