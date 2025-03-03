import numpy as np
import torch
import yaml
from tqdm import tqdm

from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
import os
import yaml
import torch
import matplotlib.pyplot as plt
from models.model import SRCNN


from utils import calculate_psnr, calculate_ssim
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss

# 加载配置
with open("config.yaml",encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 创建输出目录
os.makedirs(config["save_dir"], exist_ok=True)
os.makedirs(config["result_dir"], exist_ok=True)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRCNN().to(device)
optimizer = Adam(model.parameters(), lr=config["lr"])
criterion = MSELoss()

# 获取数据集
train_dataset, val_dataset = Dataset.get_train_val_datasets(
    config["train_lr_dir"],
    config["train_hr_dir"],
    val_split=config["val_split"]
)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 记录指标
history = {
    'train_loss': [],
    'val_loss': [],
    'psnr': [],
    'ssim': []
}

# 训练循环
epoch = 0
for epoch in tqdm(range(config["epochs"]),desc=f"Epoch {epoch+1}/{config['epochs']} [Train]"):
    # 训练阶段
    model.train()
    epoch_train_loss = 0.0
    for lr, hr in train_loader:
     #   print(lr.shape,hr.shape)
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        output = model(lr)
        loss = criterion(output, hr)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * lr.size(0)
    train_loss = epoch_train_loss / len(train_loader.dataset)
    history['train_loss'].append(train_loss)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            output = model(lr)

            # 计算损失
            val_loss += criterion(output, hr).item()

            # 转换为图像格式计算指标
            output_img = output.squeeze().cpu().numpy().transpose(1, 2, 0)
            hr_img = hr.squeeze().cpu().numpy().transpose(1, 2, 0)

            # 反归一化到 0-255
            output_img = (output_img * 255).clip(0, 255).astype(np.uint8)
            hr_img = (hr_img * 255).clip(0, 255).astype(np.uint8)

            psnr_values.append(calculate_psnr(hr_img, output_img))
            ssim_values.append(calculate_ssim(hr_img, output_img))

    # 记录验证指标
    history['val_loss'].append(val_loss / len(val_loader))
    history['psnr'].append(np.mean(psnr_values))
    history['ssim'].append(np.mean(ssim_values))

    # 打印进度
    print(f"Epoch {epoch + 1}/{config['epochs']}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"PSNR: {history['psnr'][-1]:.2f} dB | SSIM: {history['ssim'][-1]:.4f}\n")

    # 保存模型
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"{config['save_dir']}/model_epoch_{epoch + 1}.pth")

# 保存训练曲线
plt.figure(figsize=(12, 8))

# 损失曲线
plt.subplot(2, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# PSNR曲线
plt.subplot(2, 2, 2)
plt.plot(history['psnr'], label='PSNR', color='green')
plt.xlabel('Epoch')
plt.ylabel('dB')
plt.legend()

# SSIM曲线
plt.subplot(2, 2, 3)
plt.plot(history['ssim'], label='SSIM', color='red')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

plt.tight_layout()
plt.savefig(f"{config['result_dir']}/training_metrics.png")
plt.close()