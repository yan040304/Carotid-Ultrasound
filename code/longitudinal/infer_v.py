import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image, ImageOps
import cv2
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy.spatial.distance import directed_hausdorff
import pandas as pd
from scipy import ndimage

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# UNet模型定义（与训练代码保持一致）
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 输入是CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# 自定义Padding函数（与训练代码保持一致）
def pad_to_square(image, target_size=512, fill_value=0):
    """
    将图像填充为正方形，保持原始宽高比
    :param image: PIL Image对象
    :param target_size: 目标正方形尺寸
    :param fill_value: 填充值
    :return: 填充后的PIL Image对象
    """
    width, height = image.size
    
    # 计算需要添加的padding
    if width > height:
        padding_top = (width - height) // 2
        padding_bottom = width - height - padding_top
        padding = (0, padding_top, 0, padding_bottom)
    else:
        padding_left = (height - width) // 2
        padding_right = height - width - padding_left
        padding = (padding_left, 0, padding_right, 0)
    
    # 应用padding
    padded_image = ImageOps.expand(image, padding, fill=fill_value)
    
    # 调整到目标尺寸
    if target_size is not None:
        padded_image = padded_image.resize((target_size, target_size), Image.BILINEAR)
    
    return padded_image

def load_model(model_path, device, n_classes=3):
    """加载训练好的模型"""
    model = UNet(n_channels=1, n_classes=n_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, target_size=256):
    """预处理图像，与训练时保持一致"""
    # 读取图像并转换为灰度
    image = Image.open(image_path).convert('L')
    
    # 记录原始尺寸
    original_size = image.size
    
    # 应用padding
    image = pad_to_square(image, target_size, fill_value=0)
    
    # 转换为Tensor
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    
    return image_tensor, original_size

def postprocess_mask(mask_tensor, original_size):
    """后处理分割掩码，恢复原始尺寸"""
    # 将Tensor转换为numpy数组
    mask = mask_tensor.squeeze().cpu().numpy()
    
    # 将预测值转换为标签 (0, 1, 2)
    mask = mask.astype(np.uint8)
    
    # 按照要求映射标签值：1->100, 2->255
    mask_mapped = np.zeros_like(mask, dtype=np.uint8)
    mask_mapped[mask == 1] = 100  # 血管腔
    mask_mapped[mask == 2] = 255  # 血管壁
    
    # 转换为PIL图像
    mask_pil = Image.fromarray(mask_mapped)
    
    # 恢复原始尺寸（先恢复到padding后的正方形尺寸，然后裁剪）
    # 计算padding后的正方形尺寸
    if original_size[0] > original_size[1]:
        square_size = original_size[0]
    else:
        square_size = original_size[1]
    
    # 调整到正方形尺寸
    mask_pil = mask_pil.resize((square_size, square_size), Image.NEAREST)
    
    # 裁剪到原始尺寸
    if original_size[0] > original_size[1]:
        # 原始图像是宽大于高，需要裁剪上下
        top = (square_size - original_size[1]) // 2
        bottom = square_size - original_size[1] - top
        mask_pil = mask_pil.crop((0, top, original_size[0], square_size - bottom))
    else:
        # 原始图像是高大于宽，需要裁剪左右
        left = (square_size - original_size[0]) // 2
        right = square_size - original_size[0] - left
        mask_pil = mask_pil.crop((left, 0, square_size - right, original_size[1]))
    
    return mask_pil

def calculate_dice(pred_mask, gt_mask):
    """计算Dice系数"""
    if pred_mask is None or gt_mask is None:
        return None
    
    # 确保两个掩码尺寸相同
    if pred_mask.shape != gt_mask.shape:
        gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 将预测掩码和真实掩码二值化
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    # 计算交集和并集
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum()
    
    # 计算Dice系数
    dice = (2.0 * intersection) / union if union > 0 else 0.0
    
    return dice

def calculate_hd95(pred_mask, gt_mask):
    """计算95% Hausdorff距离"""
    if pred_mask is None or gt_mask is None:
        return None
    
    # 确保两个掩码尺寸相同
    if pred_mask.shape != gt_mask.shape:
        gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 将预测掩码和真实掩码二值化
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    # 获取边界点坐标
    pred_points = np.argwhere(pred_binary > 0)
    gt_points = np.argwhere(gt_binary > 0)
    
    # 如果任一掩码没有点，返回无穷大
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float('inf')
    
    # 计算两个方向的Hausdorff距离
    hd1 = directed_hausdorff(pred_points, gt_points)[0]
    hd2 = directed_hausdorff(gt_points, pred_points)[0]
    
    # 取最大值作为Hausdorff距离
    hd = max(hd1, hd2)
    
    return hd

def process_single_image(model, image_path, output_path, device, target_size=256):
    """处理单张图像并保存结果，返回预测掩码"""
    # 预处理图像
    image_tensor, original_size = preprocess_image(image_path, target_size)
    image_tensor = image_tensor.to(device)
    
    # 模型推理
    with torch.no_grad():
        output = model(image_tensor)
        # 对于多类分割，使用argmax获取预测类别
        pred_mask = torch.argmax(output, dim=1)
    
    # 后处理分割掩码
    result_mask = postprocess_mask(pred_mask, original_size)
    
    # 保存结果
    result_mask.save(output_path)
    print(f"分割结果已保存到: {output_path}")
    
    # 返回预测掩码的numpy数组形式
    return np.array(result_mask)

def process_batch_images(model, input_dir, output_dir, label_dir, device, target_size=256, file_extensions=('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
    """处理批量图像并保存结果，同时计算评估指标"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_paths = []
    for ext in file_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, f'*.{ext}')))
        image_paths.extend(glob.glob(os.path.join(input_dir, f'*.{ext.upper()}')))
    
    if not image_paths:
        print(f"在目录 {input_dir} 中没有找到图像文件")
        return []
    
    print(f"找到 {len(image_paths)} 张图像")
    
    # 准备存储结果
    results = []
    
    # 处理每张图像
    for image_path in tqdm(image_paths, desc="处理图像"):
        try:
            # 生成输出路径
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_mask{ext}")
            
            # 处理单张图像，获取预测掩码
            pred_mask = process_single_image(model, image_path, output_path, device, target_size)
            
            # 查找对应的标签文件
            label_path = os.path.join(label_dir, filename)
            if not os.path.exists(label_path):
                # 尝试其他扩展名
                found = False
                for label_ext in file_extensions:
                    label_path_alt = os.path.join(label_dir, f"{name}.{label_ext}")
                    if os.path.exists(label_path_alt):
                        label_path = label_path_alt
                        found = True
                        break
                
                if not found:
                    print(f"警告: 未找到 {filename} 对应的标签文件")
                    gt_mask = None
                else:
                    gt_mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            else:
                gt_mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            
            # 计算评估指标
            dice = calculate_dice(pred_mask, gt_mask) if gt_mask is not None else None
            hd95 = calculate_hd95(pred_mask, gt_mask) if gt_mask is not None else None
            
            # 存储结果
            results.append({
                'filename': filename,
                'dice': round(dice,2),
                'hd95': round(hd95,2)
            })
            
            print(f"处理完成: {filename} - Dice: {dice:.4f if dice else 'N/A'}, HD95: {hd95:.4f if hd95 else 'N/A'}")
            
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
    
    return results

def save_results_to_csv(results, csv_path):
    """将结果保存到CSV文件"""
    # 创建目录
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 保存到CSV
    df.to_csv(csv_path, index=False)
    print(f"评估结果已保存到: {csv_path}")

def create_visualization(image_path, mask_path, output_path):
    """创建可视化结果，显示原始图像和分割结果"""
    # 读取图像
    image = Image.open(image_path).convert('L')
    mask = Image.open(mask_path).convert('L')
    
    # 创建可视化图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示原始图像
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 显示分割结果
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('血管区域分割')
    axes[1].axis('off')
    
    # 显示叠加结果
    overlay = np.array(image.convert('RGB'))
    mask_array = np.array(mask)
    
    # 创建彩色叠加：血管腔(100)为红色，血管壁(255)为绿色
    overlay[mask_array == 100, :] = [255, 0, 0]  # 红色表示血管腔
    overlay[mask_array == 255, :] = [0, 255, 0]  # 绿色表示血管壁
    
    axes[2].imshow(overlay)
    axes[2].set_title('叠加效果')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存到: {output_path}")

def create_batch_visualizations(input_dir, mask_dir, output_dir, file_extensions=('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
    """为批量图像创建可视化结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有掩码文件
    mask_paths = []
    for ext in file_extensions:
        mask_paths.extend(glob.glob(os.path.join(mask_dir, f'*_mask.{ext}')))
        mask_paths.extend(glob.glob(os.path.join(mask_dir, f'*_mask.{ext.upper()}')))
    
    if not mask_paths:
        print(f"在目录 {mask_dir} 中没有找到掩码文件")
        return
    
    print(f"找到 {len(mask_paths)} 个掩码文件")
    
    # 为每个掩码创建可视化
    for mask_path in tqdm(mask_paths, desc="创建可视化"):
        # 找到对应的原始图像
        mask_name = os.path.basename(mask_path)
        original_name = mask_name.replace("_mask", "")
        original_path = os.path.join(input_dir, original_name)
        
        if os.path.exists(original_path):
            # 生成输出路径
            name, ext = os.path.splitext(original_name)
            output_path = os.path.join(output_dir, f"{name}_visualization.png")
            
            # 创建可视化
            try:
                create_visualization(original_path, mask_path, output_path)
            except Exception as e:
                print(f"创建可视化 {original_name} 时出错: {e}")
        else:
            print(f"找不到对应的原始图像: {original_name}")

if __name__ == "__main__":
    # === 路径配置 ===
    model_path = "code/longitudinal/carotid_unet_longitudinal.pth"
    input_dir = "data/longitudinal/image"
    label_dir = "data/longitudinal/label"  # 标签目录
    output_dir = "results/deep_learning/longitudinal"
    viz_dir = "results/deep_learning/longitudinal_visualizations"
    results_csv_path = "results/diagnosis/longitudinal/dl_evaluation_results.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print(f"加载模型: {model_path}")
    model = load_model(model_path, device, n_classes=3)

    print(f"开始推理: {input_dir}")
    # 处理批量图像并获取评估结果
    evaluation_results = process_batch_images(model, input_dir, output_dir, label_dir, device, target_size=256)
    
    # 保存评估结果到CSV
    if evaluation_results:
        save_results_to_csv(evaluation_results, results_csv_path)
    
    print(f"生成可视化结果: {viz_dir}")
    create_batch_visualizations(input_dir, output_dir, viz_dir)

    print("\n纵切深度学习推理与可视化完成！")
    print(f"评估结果已保存到: {results_csv_path}")