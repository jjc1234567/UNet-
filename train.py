import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt
import numpy as np
import shutil
from basic.unet import UNet  # 导入新的UNet定义
import pre

import glob
from mmpose.apis import MMPoseInferencer
import pandas as pd

# 新增的导入
from pytorch_msssim import ssim

# 使用模型配置文件和权重文件的路径或 URL 构建推理器
inferencer = MMPoseInferencer(
    pose2d='mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
    pose2d_weights='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
)


# 数据集类
class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.original_files = os.listdir(os.path.join(root_dir, 'LSP_Ori'))
        self.optimized_files = os.listdir(os.path.join(root_dir, 'LSP_EM'))
        self.transform = transform

    def __len__(self):
        return len(self.original_files)

    def __getitem__(self, idx):
        original_path = os.path.join(self.root_dir, 'LSP_Ori', f"Smo_{idx}.jpg")
        optimized_path = os.path.join(self.root_dir, 'LSP_EM', f"Image_{idx}.jpg")

        original_image = Image.open(original_path).convert("RGB")
        optimized_image = Image.open(optimized_path).convert("RGB")
        
        if self.transform:
            transformed_original = self.transform(original_image)
            transformed_optimized = self.transform(optimized_image)

        return {'original': transformed_original, 'optimized': transformed_optimized}


# 修改损失函数
def custom_loss(original_images, outputs):
    w1 = 0.3  # MSE 权重
    w2 = 0.3  # SSIM 权重
    w3 = 0.4  # HPE 权重

    mse = mse_loss(original_images, outputs)
    ssim_value = 1 - ssim(original_images, outputs, data_range=1.0, size_average=True)

    hpe_oks = oks_keypoints(original_images, outputs)  # 计算 OKS

    loss = w1 * mse + w2 * ssim_value + w3 * (1 - hpe_oks)
    return loss

def oks_keypoints(original_images, outputs):
    num_images = len(original_images)
    oks_ave = np.zeros(num_images)
    
    # 定义 sigmas，此处需要根据您的关键点定义调整
    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    for i in range(num_images):
        ori_image = original_images[i]
        em_image = outputs[i]

        # 将张量转换为 PIL 图像
        ori_pil_image = TF.to_pil_image(ori_image.cpu())
        em_pil_image = TF.to_pil_image(em_image.cpu())

        # 将 PIL 图像转换为 NumPy 数组
        ori_array = np.array(ori_pil_image)
        em_array = np.array(em_pil_image)

        # 获取姿态估计结果
        pose_results_ori = next(inferencer(ori_array))
        pose_results_em = next(inferencer(em_array))

        # 打印返回的结果，以检查其结构
        # print(f"Pose Results Original: {pose_results_ori}")
        # print(f"Pose Results EM: {pose_results_em}")

        # 确保从结果中提取关键点
        if ('predictions' in pose_results_ori and
            len(pose_results_ori['predictions']) > 0 and
            len(pose_results_ori['predictions'][0]) > 0):
            
            predictions_ori = pose_results_ori['predictions'][0][0]
            keypoints_truth_xy = np.array(predictions_ori['keypoints'])
            keypoints_truth_score = np.array(predictions_ori['keypoint_scores']).reshape(-1, 1)
            keypoints_truth = np.hstack((keypoints_truth_xy, keypoints_truth_score))
        else:
            print("No valid predictions in original image.")
            continue

        if ('predictions' in pose_results_em and
            len(pose_results_em['predictions']) > 0 and
            len(pose_results_em['predictions'][0]) > 0):
            
            predictions_em = pose_results_em['predictions'][0][0]
            keypoints_pred_xy = np.array(predictions_em['keypoints'])
            keypoints_pred_score = np.array(predictions_em['keypoint_scores']).reshape(-1, 1)
            keypoints_pred = np.hstack((keypoints_pred_xy, keypoints_pred_score))
        else:
            print("No valid predictions in enhanced image.")
            continue

        # 检查关键点的形状
        if keypoints_truth.shape[1] != 3 or keypoints_pred.shape[1] != 3:
            print("Warning: keypoints array does not contain (x, y, score).")
            continue

        # print(f"Keypoints Truth Shape: {keypoints_truth.shape}")  # 调试信息
        # print(f"Keypoints Pred Shape: {keypoints_pred.shape}")    # 调试信息

        # 计算 OKS
        num_keypoints = keypoints_truth.shape[0]

        min_x = np.min(keypoints_truth[:, 0])
        max_x = np.max(keypoints_truth[:, 0])
        min_y = np.min(keypoints_truth[:, 1])
        max_y = np.max(keypoints_truth[:, 1])

        width = max_x - min_x
        height = max_y - min_y
        s = np.sqrt(width**2 + height**2)  # bounding box 的对角线长度
        oks = np.zeros(num_keypoints)

        # 计算置信度得分大于等于0.5的关键点数量
        above_threshold_count_body = np.sum(keypoints_truth[:, 2] >= 0.5)

        for j in range(num_keypoints):
            x_truth, y_truth, score_truth = keypoints_truth[j]
            x_pred, y_pred, _ = keypoints_pred[j]

            if score_truth >= 0.5:
                distance = np.sqrt((x_truth - x_pred) ** 2 + (y_truth - y_pred) ** 2)
                e = distance ** 2 / (2 * (s * sigmas[j]) ** 2)
                oks[j] = np.exp(-e)

        if above_threshold_count_body == 0:
            oks_ave_body = 1.0
        else:
            oks_ave_body = np.sum(oks) / above_threshold_count_body

        oks_ave[i] = oks_ave_body

    oks_final = np.sum(oks_ave) / num_images

    print(f"OKS Ave: {oks_final}")
    return oks_final


# 超参数
batch_size = 32
lr = 1e-4
num_epochs = 500

# 数据预处理和转换
transform = transforms.Compose([
    transforms.Resize((300, 300)),  
    transforms.ToTensor(),
])

dataset = MyDataset(root_dir='dataset', transform=transform)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

train_indices = list(range(train_size))
test_indices = list(range(train_size, len(dataset)))

train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 保存测试集原始图片
original_test_folder = 'original_test_images'
os.makedirs(original_test_folder, exist_ok=True)
for i, batch in enumerate(test_loader):
    original_images = batch['original']
    original_file_name = test_dataset.dataset.original_files[i]  # 修改此行
    original_image_path = os.path.join('dataset/LSP_Ori', original_file_name)
    target_image_path = os.path.join(original_test_folder, original_file_name)
    shutil.copy(original_image_path, target_image_path)

# 初始化模型、损失函数和优化器
model = UNet(in_channels=3, out_channels=3)  
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

# 将模型和数据移动到 GPU 上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 训练模型
train_loss_logger = []  # 保存训练损失
test_loss_logger = []

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        original_images = batch['original'].to(device)
        optimized_images = batch['optimized'].to(device)

        optimizer.zero_grad()
        outputs = model(optimized_images)
        outputs = torch.nn.functional.interpolate(outputs, size=(original_images.size(2), original_images.size(3)), mode='bilinear', align_corners=False)
       
        loss = custom_loss(original_images, outputs)
        loss.backward()
        optimizer.step()

        train_loss_logger.append(loss.item())  # 记录训练损失

    scheduler.step()  
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    # 在训练循环中每隔5次迭代计算测试集损失
    if (epoch + 1) % 5 == 0:
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                original_images = batch['original'].to(device)
                optimized_images = batch['optimized'].to(device)

                outputs = model(optimized_images)
                outputs = torch.nn.functional.interpolate(outputs, size=(original_images.size(2), original_images.size(3)), mode='bilinear', align_corners=False)

                loss = custom_loss(original_images, outputs)
                test_loss += loss.item() * original_images.size(0)

        test_loss /= len(test_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss}')
        test_loss_logger.append(test_loss)

    # 每50个epoch调用optimize_images进行预测
    if (epoch + 1) % 5 == 0:
        # 保存模型的状态
        torch.save(model.state_dict(), f'larger_unet_lsp_epoch_{epoch + 1}.pth')

        # 使用optimize_images方法进行预测
        pre.optimize_images(input_folder='original_test_images', model_path=f'larger_unet_lsp_epoch_{epoch + 1}.pth', output_folder=f'optimized_results_epoch_{epoch + 1}')

# 保存模型
torch.save(model.state_dict(), 'larger_unet_model_LSP.pth')

train_loss_array = np.array(train_loss_logger)
test_loss_array = np.array(test_loss_logger)

# 生成损失曲线图
plt.plot(train_loss_array, label='Train Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.savefig('train_loss_curve.png')

plt.plot(test_loss_array, label='Test Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.savefig('test_loss_curve.png')
