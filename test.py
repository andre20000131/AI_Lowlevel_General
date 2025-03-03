import torch
import yaml
from models.model import SRCNN
from datasets import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np

# 加载配置和模型
with open("config.yaml",encoding='utf-8') as f:
    config = yaml.safe_load(f)

model = SRCNN()
model.load_state_dict(torch.load("./train_res/checkpoints/model_epoch_10.pth",weights_only=True))
model.eval()

# 测试数据集
test_dataset = Dataset(config["val_lr_dir"], config["val_hr_dir"], patch_size=None)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 测试并保存结果
with torch.no_grad():
    for i, (lr, hr) in enumerate(test_loader):
        output = model(lr)
        output = output.squeeze().permute(1, 2, 0).numpy() * 255
        output = output.astype(np.uint8)
        cv2.imwrite(f"./test_res/output_{i}.png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))