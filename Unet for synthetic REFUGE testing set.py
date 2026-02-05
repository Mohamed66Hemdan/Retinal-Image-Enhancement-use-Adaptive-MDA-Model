import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import os
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import cv2
import random
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# ============================================================
# 1. CONFIGURATION
# ============================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 256
IMG_DIR = "/content/REFUGE/train/Images"
MASK_DIR = "/content/REFUGE/train/gts"
ENHANCER_WEIGHTS = "/content/adaptive_mda_net_final (2).pth"
NUM_CLASSES = 3  # number of segmentation classes ( background, optic disc, cup)

# ============================================================
# 2. MDA-NET (IMAGE ENHANCEMENT)
# ============================================================

# Channel Attention: emphasizes important channels in features
class ChannelAttention(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c//reduction, c, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# Spatial Attention: emphasizes important spatial regions
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, 1, 3)
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return x * torch.sigmoid(self.conv(y))

# Degradation Analyzer: encodes the degradation level of input image
class DegradationAnalyzer(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.analyze = nn.Sequential(
            nn.Linear(input_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, latent_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pool(x).view(x.size(0), -1)
        return self.analyze(y).view(x.size(0), -1, 1, 1)

# Dynamic Residual Block: combines convolution, spatial & channel attention
class DynamicResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.InstanceNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.InstanceNorm2d(c)
        )
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention()
        self.modulation = nn.Sequential(nn.Conv2d(256, c, 1), nn.Sigmoid())
    def forward(self, x, lv):
        out = self.conv(x) * self.modulation(lv)
        return x + self.sa(self.ca(out))

# Adaptive Generator (MDA-Net): Enhances degraded retinal images
class AdaptiveGenerator(nn.Module):
    """
    Enhances degraded retinal images using dynamic residual blocks
    guided by degradation analysis.
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
# Apply synthetic degradation to images
def degrade_image_mdanet(img_pil, level='mid'):
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

# Convert enhanced tensor to clean image for visualization
def clean_enhanced_image(img_tensor):
    img_np = ((img_tensor.detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    img_cleaned = cv2.medianBlur(img_np, 5)
    return img_cleaned / 255.0

# Compute Dice and IoU metrics for segmentation
def calculate_metrics(pred, gt):
    dice_list, iou_list = [], []
    for i in range(1, 3):  # ignore background
        p, g = (pred == i).astype(np.float32), (gt == i).astype(np.float32)
        inter = np.sum(p * g)
        union = np.sum(p) + np.sum(g)
        dice_list.append((2. * inter) / (union + 1e-7))
        iou_list.append(inter / (union - inter + 1e-7))
    return np.mean(dice_list), np.mean(iou_list)

# Custom Dataset for REFUGE (original + degraded + GT masks)
class REFUGEJointDataset(Dataset):
    """
    Returns tuple: (original_image, degraded_image, ground_truth_mask)
    """
    def __init__(self, img_dir, mask_dir):
        self.imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        self.img_dir, self.mask_dir = img_dir, mask_dir
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        base_name = os.path.splitext(img_name)[0]
        mask_path = next((os.path.join(self.mask_dir, base_name + ext) for ext in ['.png', '.bmp', '.jpg'] if os.path.exists(os.path.join(self.mask_dir, base_name + ext))), None)
        img_pil = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        mask_pil = Image.open(mask_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
        mask_np = np.array(mask_pil)
        final_mask = np.zeros_like(mask_np)
        final_mask[mask_np < 50] = 2  # class 2
        final_mask[(mask_np >= 50) & (mask_np < 200)] = 1  # class 1
        return transforms.ToTensor()(img_pil.resize((IMG_SIZE, IMG_SIZE))), degrade_image_mdanet(img_pil), torch.tensor(final_mask).long()

# ============================================================
# 4. MAIN EXECUTION & VISUALIZATION
# ============================================================

def main():
    # Load MDA-Net enhancer
    enhancer = AdaptiveGenerator().to(DEVICE)
    if os.path.exists(ENHANCER_WEIGHTS):
        enhancer.load_state_dict(torch.load(ENHANCER_WEIGHTS, map_location=DEVICE))
    enhancer.eval()

    # Segmentation model (U-Net with ResNet34 encoder pretrained on ImageNet)
    seg_model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES).to(DEVICE)
    # TODO: load pre-trained segmentation weights if available

    dataset = REFUGEJointDataset(IMG_DIR, MASK_DIR)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = optim.Adam(seg_model.parameters(), lr=2e-4)
    print("Training Segmentation Model (Quick Demo)...")
    for epoch in range(10):
        seg_model.train()
        for _, deg, masks in loader:
            deg, masks = deg.to(DEVICE), masks.to(DEVICE)
            with torch.no_grad(): enhanced = torch.clamp(enhancer(deg), -1, 1)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(seg_model(enhanced), masks)
            loss.backward(); optimizer.step()
        print(f"Epoch {epoch+1}/10 completed.")

    # Visualize 15 random samples
    indices = random.sample(range(len(dataset)), 15)
    for count, idx in enumerate(indices):
        orig, deg, gt = dataset[idx]
        with torch.no_grad():
            enhanced_raw = enhancer(deg.unsqueeze(0).to(DEVICE))[0]
            enhanced_img = clean_enhanced_image(enhanced_raw)

            pred_deg_logits = seg_model(deg.unsqueeze(0).to(DEVICE))
            pred_deg = torch.argmax(F.softmax(pred_deg_logits, dim=1), dim=1)[0].cpu().numpy()

            enh_tensor = torch.tensor(enhanced_img).permute(2,0,1).unsqueeze(0).float().to(DEVICE) * 2 - 1
            pred_enh_logits = seg_model(enh_tensor)
            pred_enh = torch.argmax(F.softmax(pred_enh_logits, dim=1), dim=1)[0].cpu().numpy()

        dice_v, iou_v = calculate_metrics(pred_enh, gt.numpy())

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Sample {count+1}/15 | Dice: {dice_v:.4f} | IoU: {iou_v:.4f}", fontsize=16, fontweight='bold')

        axes[0, 0].imshow(orig.permute(1,2,0)); axes[0, 0].set_title("Original Image")
        axes[0, 1].imshow(((deg.permute(1,2,0)+1)/2).clip(0,1)); axes[0, 1].set_title("Degraded Input")
        axes[0, 2].imshow(enhanced_img); axes[0, 2].set_title("Cleaned Enhanced Output")
        axes[1, 0].imshow(gt, cmap='gray', vmin=0, vmax=2); axes[1, 0].set_title("Ground Truth Mask")
        axes[1, 1].imshow(pred_deg, cmap='gray', vmin=0, vmax=2); axes[1, 1].set_title("Pred on Degraded")
        axes[1, 2].imshow(pred_enh, cmap='gray', vmin=0, vmax=2); axes[1, 2].set_title("Pred on Enhanced")
        for ax in axes.ravel(): ax.axis("off")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == "__main__":
    main()
