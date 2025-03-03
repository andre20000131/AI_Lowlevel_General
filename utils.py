import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    # print(img1.shape)
    # print(img2.shape)
    return psnr(img1, img2, data_range=255)

def calculate_ssim(img1, img2):
    # print(1111)
    # print(img1.shape)
    # print(img2.shape)
    return ssim(img1, img2, channel_axis=2,data_range=255)