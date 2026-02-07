# ============================================================
# Adaptive MDA-Net 
# ============================================================

# =========================
# Libraries
# =========================
import os, random, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# =========================
# Config
# =========================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 200
LR = 2e-4
RAW_DATA = '/content/d1/d1'
CSV_PATH = '/content/Label_EyeQ_train.csv'
SPLIT_ROOT = '/content/split_data'
SAVE_PATH = 'adaptive_mda_net.pth'
FIQA_PATH = 'fiqa_trained.pth'
SHOW_TRAIN_IMAGES = True

# ============================================================
# Dataset Splitting (EyeQ protocol)
# ============================================================
quality_map = {0: 'Good', 1: 'Usable', 2: 'Reject'}
for v in quality_map.values():
    os.makedirs(os.path.join(SPLIT_ROOT, v), exist_ok=True)

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    df['base'] = df['image'].astype(str).str.strip().apply(lambda x: os.path.splitext(x)[0])
    base2quality = dict(zip(df['base'], df['quality']))

    for img in os.listdir(RAW_DATA):
        if not img.lower().endswith(('.jpg','.png','.jpeg')):
            continue
        base_img = os.path.splitext(img)[0].replace('-600','')
        cls = quality_map.get(base2quality.get(base_img, 0), 'Good')
        shutil.copy(os.path.join(RAW_DATA, img), os.path.join(SPLIT_ROOT, cls, img))


# ============================================================
# Image Degradation Function for MDA-Net
# ============================================================
# This function simulates real-world degradations commonly found in retinal fundus images.
# Degradations applied:
    # 1. Resize: Resize input image to a fixed size (IMG_SIZE x IMG_SIZE)
    # 2. Vignette: Darken image edges to mimic uneven illumination in real retinal images
    # 3. Gamma Correction: Adjust contrast non-linearly to simulate exposure differences
    # 4. Brightness Adjustment: Multiply image by a factor to simulate under/overexposed images
    # 5. Gaussian Blur: Simulate slight out-of-focus or motion blur
    # 6. Noise Addition: Simulate sensor noise or low-light acquisition noise
    # 7. Color Shift: Randomly adjust RGB channels to mimic device-dependent color variations
    # 8. Mixing Blur and Noise: Combine blurry and noisy versions for more realistic degradations
    # 9. Output Scaling: Transform image from [0,1] to [-1,1] for stable training (commonly used with tanh activation)
#
# Usage:
# 'level' can be 'low', 'mid', or 'high' to control severity of degradations

def degrade_image_mdanet(img_pil, level='mid'):
    """Simulate realistic retinal degradations (illumination, blur, noise)."""
    img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE))
    img = transforms.ToTensor()(img_pil)

    levels = {
        'low':  {'gamma':0.8,'brightness':0.9,'blur':3,'noise':0.01},
        'mid':  {'gamma':1.2,'brightness':0.75,'blur':7,'noise':0.03},
        'high': {'gamma':1.6,'brightness':0.6,'blur':11,'noise':0.05},
    }
    p = levels[level]

    # Vignette (uneven illumination)
    h, w = IMG_SIZE, IMG_SIZE
    y, x = torch.meshgrid(torch.linspace(-1,1,h), torch.linspace(-1,1,w), indexing='ij')
    vignette = 1 - 0.5 * (x**2 + y**2)
    vignette = vignette.clamp(0.4,1.0)
    img = img * vignette

    # Gamma + brightness
    img = (img ** p['gamma']) * p['brightness']
    img = img.clamp(0,1)

    # Blur
    blur_img = img_pil.filter(ImageFilter.GaussianBlur(radius=p['blur']))
    blur_img = transforms.ToTensor()(blur_img)

    # Noise
    noise = torch.randn_like(img) * p['noise']
    noisy = (img + noise).clamp(0,1)

    # Color shift
    color_shift = torch.tensor([
        np.random.uniform(0.9,1.1),
        np.random.uniform(0.8,1.2),
        np.random.uniform(0.9,1.1)
    ]).view(3,1,1)
    noisy = (noisy * color_shift).clamp(0,1)

    degraded = 0.6 * noisy + 0.4 * blur_img
    return degraded * 2 - 1   # [-1,1]



# ============================================================
# Fundus Dataset with Integrated Degradation for MDA-Net
# ============================================================
# This Dataset class performs both preprocessing and task-specific data augmentation for training an image enhancement network.
# 
# Preprocessing includes:
# 1. Resize: Ensures all images have the same dimensions (IMG_SIZE x IMG_SIZE)
# 2. ToTensor: Converts PIL image to PyTorch Tensor in [0,1]
# 3. Normalize: Scales tensor values to [-1,1] for stable training (suitable for tanh output)
#
# Degradation / Augmentation includes:
# 1. Vignette: Darkens edges to simulate uneven illumination
# 2. Gamma correction: Adjusts contrast non-linearly
# 3. Brightness scaling: Simulates under/overexposed images
# 4. Gaussian blur: Simulates slight defocus or motion blur
# 5. Additive noise: Simulates sensor / low-light noise
# 6. Color shift: Random RGB imbalance per image
# 7. Random degradation levels: 'low', 'mid', 'high' to increase variability

class FundusDataset(Dataset):
    def __init__(self, root):
        self.paths = [os.path.join(root, x) for x in os.listdir(root)]
        self.tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        level = random.choice(['low','mid','high'])
        Il = degrade_image_mdanet(img, level)   # degraded input
        Ih = self.tf(img)                       # clean target
        return Il, Ih, self.paths[idx]

# ========================= ====================================================================================================================
# ========================= ======================================== Build Model ======================================== ======================
# ========================= ============================================================================================== =====================
# Attention & Adaptive Modules
# ============================================================
# ============================================================
# Channel Attention 
# ============================================================
# Learns to emphasize important feature channels by generating a channel-wise attention map.
# Uses global average pooling followed by a small MLP to produce channel weights.

class ChannelAttention(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c//reduction), nn.ReLU(), nn.Linear(c//reduction, c), nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_ = x.size()
        y = self.fc(self.avg(x).view(b,c)).view(b,c,1,1)
        return x*y

# ============================================================
# Spatial Attention 
# ============================================================
# Learns to emphasize important spatial locations by generating a spatial attention map.
# Combines channel-wise max and average pooling, followed by a convolution and sigmoid activation.

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2,1,7,1,3)
    def forward(self,x):
        avg = x.mean(1,True)
        mx  = x.max(1,True)[0]
        y = torch.cat([avg,mx],1)
        return x * torch.sigmoid(self.conv(y))

# ============================================================
# Degradation Analyzer 
# ============================================================
# Computes a latent vector representing the type and level of image degradation.
# Uses global average pooling and a small MLP to generate per-channel latent codes.
class DegradationAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(3,64), nn.ReLU(), nn.Linear(64,256), nn.Sigmoid())
    def forward(self,x):
        b,c,_,_ = x.size()
        return self.fc(self.pool(x).view(b,c)).view(b,-1,1,1)

# ============================================================
# Dynamic Residual Block with Attention
# ============================================================
# Residual block enhanced with channel and spatial attention, modulated by degradation latent vector.
# Learns to adapt feature refinement dynamically based on estimated image degradation.
class DynamicResBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c,c,3,1,1), nn.InstanceNorm2d(c), nn.ReLU(),
            nn.Conv2d(c,c,3,1,1), nn.InstanceNorm2d(c)
        )
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention()
        self.mod = nn.Sequential(nn.Conv2d(256,c,1), nn.Sigmoid())
    def forward(self,x,lv):
        out = self.conv(x) * self.mod(lv)
        out = self.ca(out)
        out = self.sa(out)
        return x + out
# ============================================================
# Adaptive Generator (MDA-Net Style, Improved)
# ============================================================
# Generates enhanced retinal images from degraded inputs using an adaptive, multi-stage architecture.
# Extracts a latent degradation vector with DegradationAnalyzer to modulate feature refinement.
# Encodes input via three convolutional layers, processes with multiple DynamicResBlocks (attention + modulation).
# Decodes features through transposed convolutions with skip connections for coarse reconstruction.
# Applies two-stage refinement (guided by estimated illumination) to produce high-quality, artifact-free output.
class AdaptiveGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = DegradationAnalyzer()
        self.enc1 = nn.Conv2d(3,64,4,2,1)
        self.enc2 = nn.Conv2d(64,128,4,2,1)
        self.enc3 = nn.Conv2d(128,256,4,2,1)
        self.blocks = nn.ModuleList([DynamicResBlock(256) for _ in range(4)])
        self.dec1 = nn.ConvTranspose2d(256,128,4,2,1)
        self.dec2 = nn.ConvTranspose2d(128,64,4,2,1)
        self.dec3 = nn.ConvTranspose2d(64,3,4,2,1)
    def forward(self,x):
        lv = self.analyzer(x)
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        f = e3
        for b in self.blocks:
            f = b(f,lv)
        d1 = F.relu(self.dec1(f)) + e2
        d2 = F.relu(self.dec2(d1)) + e1
        return torch.tanh(self.dec3(d2))


# ============================================================
# Losses (Added FFT Frequency Loss)
# ============================================================
# Measures perceptual similarity between predicted and target images using VGG16 features.
# Encourages the network to preserve high-level textures and content rather than just pixel-wise accuracy.
# Encourages the network to preserve image edges by comparing gradients.
# Computes Sobel gradients along horizontal direction on the mean of RGB channels.
# Encourages the network to match the frequency content of the target image.
# Computes the 2D Fourier Transform of both predicted and ground-truth images.
# ============================================================
# VGG Perceptual Loss
# ============================================================

class VGGPerceptual(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights='DEFAULT').features[:16].eval().to(DEVICE)
        for p in vgg.parameters(): p.requires_grad=False
        self.vgg = vgg
    def forward(self,x,y): return F.l1_loss(self.vgg(x), self.vgg(y))

def edge_loss(pred,gt):
    sobel = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float).view(1,1,3,3).to(DEVICE)
    return F.l1_loss(F.conv2d(pred.mean(1,True),sobel,padding=1), F.conv2d(gt.mean(1,True),sobel,padding=1))

def fft_loss(pred,gt):
    return F.l1_loss(torch.abs(torch.fft.fft2(pred)), torch.abs(torch.fft.fft2(gt)))

# ============================================================
# Metrics
# ============================================================
def calc_metrics(pred, gt):
    pred_np = np.clip(pred.cpu().numpy().transpose(1,2,0)*0.5+0.5,0,1)
    gt_np   = np.clip(gt.cpu().numpy().transpose(1,2,0)*0.5+0.5,0,1)
    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=1)
    ssim = structural_similarity(gt_np, pred_np, channel_axis=2, data_range=1)
    return psnr, ssim

# ============================================================
# FIQA: Training + Evaluation 
# ============================================================
class EyeQ_FIQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet50(weights='DEFAULT')
        self.net.fc = nn.Linear(2048,3)
    def forward(self,x): return self.net(x)

class EyeQQualityDataset(Dataset):
    def __init__(self, root, csv):
        self.df = pd.read_csv(csv)
        self.root = root
        self.tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        img = Image.open(os.path.join(self.root, self.df.iloc[i]['image'])).convert('RGB')
        return self.tf(img), int(self.df.iloc[i]['quality'])

# ---- Train FIQA ONCE ----
def train_fiqa():
    ds = EyeQQualityDataset(RAW_DATA, CSV_PATH)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    model = EyeQ_FIQA().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(),1e-4)
    ce = nn.CrossEntropyLoss()
    for _ in range(5):
        for x,y in dl:
            x,y = x.to(DEVICE), y.to(DEVICE)
            loss = ce(model(x),y)
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save(model.state_dict(), FIQA_PATH)

# ============================================================
# WFQA 
# ============================================================
def compute_wfqa(before_p, after_p):
    weight = {0:1.0, 1:0.6, 2:0.2}
    b = before_p.argmax()
    a = after_p.argmax()
    return max(weight[b] - weight[a], 0)

# ============================================================
# Training 
# ============================================================
def train():
    train_ds = FundusDataset(os.path.join(SPLIT_ROOT,'Good'))
    usable_ds = FundusDataset(os.path.join(SPLIT_ROOT,'Usable'))
    reject_ds = FundusDataset(os.path.join(SPLIT_ROOT,'Reject'))

    train_dl = DataLoader(train_ds,BATCH_SIZE,shuffle=True)
    usable_dl = DataLoader(usable_ds,1,shuffle=False)
    reject_dl = DataLoader(reject_ds,1,shuffle=False)

    model = AdaptiveGenerator().to(DEVICE)
    vgg_loss_fn = VGGPerceptual()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
    l1 = nn.L1Loss()

    for epoch in range(1,EPOCHS+1):
        model.train()
        for i,(Il,Ih,_) in enumerate(train_dl):
            Il,Ih = Il.to(DEVICE), Ih.to(DEVICE)
            out = model(Il)
            loss = l1(out,Ih) + 0.3*vgg_loss_fn(out,Ih) + 0.2*edge_loss(out,Ih) + 0.1*fft_loss(out,Ih)
            opt.zero_grad(); loss.backward(); opt.step()

            if i==len(train_dl)-1 and SHOW_TRAIN_IMAGES:
                show_images(Ih[0],Il[0],out[0],f' | Epoch {epoch}')

        sched.step()

        # Evaluation FIQA/WFQA 
        model.eval()
        psnr_list, ssim_list = [], []
        fiqa_usable, wfqa_usable = [], []
        fiqa_reject, wfqa_reject = [], []

        with torch.no_grad():
            for Il,Ih,_ in usable_dl:
                restored = model(Il.to(DEVICE))[0]
                p,s = calc_metrics(restored, Ih[0].to(DEVICE))
                psnr_list.append(p); ssim_list.append(s)

                before = predict_quality(tensor_to_fiqa_input(Il[0]))
                after  = predict_quality(tensor_to_fiqa_input(restored))
                f = compute_fiqa(before, after)
                w = compute_wfqa(f, after)
                fiqa_usable.append(f)
                wfqa_usable.append(w)

            for Il,Ih,_ in reject_dl:
                restored = model(Il.to(DEVICE))[0]
                p,s = calc_metrics(restored, Ih[0].to(DEVICE))
                psnr_list.append(p); ssim_list.append(s)

                before = predict_quality(tensor_to_fiqa_input(Il[0]))
                after  = predict_quality(tensor_to_fiqa_input(restored))
                f = compute_fiqa(before, after)
                w = compute_wfqa(f, after)
                fiqa_reject.append(f)
                wfqa_reject.append(w)

        FIQA_usable = np.mean(fiqa_usable) if fiqa_usable else 0
        FIQA_reject = np.mean(fiqa_reject) if fiqa_reject else 0
        WFQA_usable = np.mean(wfqa_usable) if wfqa_usable else 0
        WFQA_reject = np.mean(wfqa_reject) if wfqa_reject else 0
        WFQA_total  = 0.7*WFQA_usable + 0.3*WFQA_reject

        print(f"Epoch [{epoch}/{EPOCHS}] | PSNR: {np.mean(psnr_list):.2f} | SSIM: {np.mean(ssim_list):.4f}")
        print(f"FIQA >> Usable: {FIQA_usable:.4f} | Reject: {FIQA_reject:.4f}")
        print(f"WFQA >> Usable: {WFQA_usable:.4f} | Reject: {WFQA_reject:.4f} | Total: {WFQA_total:.4f}")
        print("-"*60)

        torch.save(model.state_dict(), SAVE_PATH)

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    train()













