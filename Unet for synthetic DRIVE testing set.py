import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# ============================================================
# 1. CONFIGURATION
# ============================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 256
IMG_DIR = "/content/files/DRIVE/training/images"
MASK_DIR = "/content/files/DRIVE/training/mask"
ENHANCER_WEIGHTS = "/content/adaptive_mda_net_final.pth"
SAVE_PATH = "vessel_segmentation_model.pth"

# ============================================================
# 2. ADAPTIVE MDA-NET COMPONENTS
# ============================================================

# Channel Attention Module: emphasizes important channels in feature maps
class ChannelAttention(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c//reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(c//reduction, c, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# Spatial Attention Module: emphasizes important spatial regions
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, 1, 3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return x * torch.sigmoid(self.conv(y))

# Degradation Analyzer: encodes image degradation information
class DegradationAnalyzer(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.analyze = nn.Sequential(
            nn.Linear(input_channels, 64), nn.ReLU(inplace=True),
            nn.Linear(64, latent_dim), nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pool(x).view(x.size(0), -1)
        return self.analyze(y).view(x.size(0), -1, 1, 1)

# Dynamic Residual Block: combines spatial & channel attention with modulation
class DynamicResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1), nn.InstanceNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1), nn.InstanceNorm2d(c)
        )
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention()
        self.modulation = nn.Sequential(nn.Conv2d(256, c, 1), nn.Sigmoid())

    def forward(self, x, lv):
        out = self.conv(x) * self.modulation(lv)
        return x + self.sa(self.ca(out))

# Adaptive Generator: main image enhancement network
class AdaptiveGenerator(nn.Module):
    """
    Enhances degraded retinal images using dynamic residual blocks
    guided by the degradation analyzer. Produces refined high-quality images.
    """
    def __init__(self):
        super().__init__()
        self.analyzer = DegradationAnalyzer()
        self.enc1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.enc2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.enc3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.light_branch = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid()
        )
        self.res_blocks = nn.ModuleList([DynamicResBlock(256) for _ in range(4)])
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec3 = nn.ConvTranspose2d(64, 3, 4, 2, 1)
        self.refine_stage1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1), nn.Tanh()
        )
        self.refine_stage2 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh()
        )

    def forward(self, x):
        lv = self.analyzer(x)
        light = self.light_branch(x)
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        f = e3
        for blk in self.res_blocks:
            f = blk(f, lv)
        d1 = F.relu(self.dec1(f)) + e2
        d2 = F.relu(self.dec2(d1)) + e1
        coarse = torch.tanh(self.dec3(d2))
        r1 = self.refine_stage1(coarse * (light + 1))
        return self.refine_stage2(r1)

# ============================================================
# 3. DATA AUGMENTATION, LOSS, AND DATASET
# ============================================================

# Apply synthetic degradation to images
def apply_degrade(img_pil, level='mid'):
    # Resize to fixed size
    img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE))
    img = transforms.ToTensor()(img_pil)  # Convert to tensor in [0,1]

    # Define degradation levels
    levels = {
        'low':  {'gamma':0.8,'brightness':0.9,'blur':3,'noise':0.01},
        'mid':  {'gamma':1.2,'brightness':0.75,'blur':7,'noise':0.03},
        'high': {'gamma':1.6,'brightness':0.6,'blur':11,'noise':0.05},
    }
    p = levels[level]

    # Apply vignette to simulate darker edges
    h, w = IMG_SIZE, IMG_SIZE
    y, x = torch.meshgrid(torch.linspace(-1,1,h), torch.linspace(-1,1,w), indexing='ij')
    vignette = 1 - 0.5 * (x**2 + y**2)
    vignette = vignette.clamp(0.4,1.0)
    img = img * vignette

    # Apply gamma correction and brightness adjustment
    img = img ** p['gamma']
    img = img * p['brightness']
    img = img.clamp(0,1)

    # Create blurred version
    blur_img = img_pil.filter(ImageFilter.GaussianBlur(radius=p['blur']))
    blur_img = transforms.ToTensor()(blur_img)

    # Add random noise
    noise = torch.randn_like(img) * p['noise']
    noisy = (img + noise).clamp(0,1)

    # Random color shift per channel
    color_shift = torch.tensor([
        np.random.uniform(0.9,1.1),
        np.random.uniform(0.8,1.2),
        np.random.uniform(0.9,1.1)
    ]).view(3,1,1)
    noisy = (noisy * color_shift).clamp(0,1)

    # Combine noisy and blurred images for realistic degradation
    degraded = 0.6 * noisy + 0.4 * blur_img

    # Scale to [-1,1] for network input stability
    return degraded * 2 - 1
# Dice + BCE combined loss
class DiceBCELoss(nn.Module):
    """
    Combines Dice loss and Binary Cross-Entropy loss
    for vessel segmentation tasks.
    """
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return BCE + dice

# Dataset for DR images + degraded + GT masks
class JointDRIVEDataset(Dataset):
    """
    Returns original image, degraded image, and ground-truth mask.
    Applies horizontal flip augmentation randomly.
    """
    def __init__(self, img_dir, mask_dir, augment=True):
        self.imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(('.tif', '.png', '.jpg'))])
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.augment = augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        name = self.imgs[idx]
        img_pil = Image.open(os.path.join(self.img_dir, name)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        mask_pil = Image.open(os.path.join(self.mask_dir, f"{name.split('_')[0]}_manual1.gif")).convert("L").resize((IMG_SIZE, IMG_SIZE))
        if self.augment and random.random() > 0.5:
            img_pil, mask_pil = TF.hflip(img_pil), TF.hflip(mask_pil)
        return transforms.ToTensor()(img_pil), apply_degrade(img_pil), transforms.ToTensor()(mask_pil)

# ============================================================
# 4. MAIN TRAINING AND VISUALIZATION FUNCTION
# ============================================================

def main():
    # Load the pretrained Adaptive Generator (MDA-Net)
    enhancer = AdaptiveGenerator().to(DEVICE)
    if os.path.exists(ENHANCER_WEIGHTS):
        enhancer.load_state_dict(torch.load(ENHANCER_WEIGHTS, map_location=DEVICE))
        print("MDA-Net Weights Loaded.")
    enhancer.eval()

    # Initialize segmentation model (U-Net)
    """
    Segmentation Model (smp.Unet):
    - Encoder: ResNet34 pretrained on ImageNet
    - Decoder: Upsamples encoder features to original size
    - Input channels: 3, Output channels: 1 (binary vessel mask)
    - Uses skip connections between encoder and decoder
    """
    seg_model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)
    optimizer = optim.Adam(seg_model.parameters(), lr=1e-4)
    criterion = DiceBCELoss()

    loader = DataLoader(JointDRIVEDataset(IMG_DIR, MASK_DIR, augment=True), batch_size=4, shuffle=True)

    print("Training Segmentation (200 Epochs)...")
    for epoch in range(200):
        seg_model.train()
        for _, deg, masks in loader:
            deg, masks = deg.to(DEVICE), masks.to(DEVICE)
            with torch.no_grad():
                enhanced = torch.clamp((enhancer(deg) + 1)/2, 0, 1)
            optimizer.zero_grad()
            loss = criterion(seg_model(enhanced), masks)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/200 Completed.")

    # Save trained segmentation model
    torch.save(seg_model.state_dict(), SAVE_PATH)
    print(f"Model saved as {SAVE_PATH}")

    # Visualize 10 random examples: original, degraded, enhanced, GT, seg on degraded, seg on enhanced
    seg_model.eval()
    eval_loader = DataLoader(JointDRIVEDataset(IMG_DIR, MASK_DIR, augment=False), batch_size=10, shuffle=True)
    normal, deg, masks = next(iter(eval_loader))

    with torch.no_grad():
        deg_device = deg.to(DEVICE)
        enhanced = torch.clamp((enhancer(deg_device) + 1)/2, 0, 1)
        pred_enh = (torch.sigmoid(seg_model(enhanced)) > 0.5).float()
        pred_deg = (torch.sigmoid(seg_model(torch.clamp((deg_device+1)/2, 0, 1))) > 0.5).float()

    fig, axes = plt.subplots(10, 6, figsize=(25, 45))
    titles = ["1. Original", "2. Degraded", "3. Enhanced", "4. GT Mask", "5. Seg on Degraded", "6. Seg on Enhanced"]

    for i in range(10):
        images = [normal[i], (deg[i]+1)/2, enhanced[i].cpu(), masks[i], pred_deg[i].cpu(), pred_enh[i].cpu()]
        for j in range(6):
            ax = axes[i, j]
            curr_img = images[j].permute(1,2,0).squeeze() if images[j].dim()==3 else images[j].squeeze()
            ax.imshow(curr_img.clip(0,1), cmap='gray' if j >= 3 else None)
            if i == 0:
                ax.set_title(titles[j], fontsize=14, fontweight='bold')
            ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
