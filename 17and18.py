# U-Net 實作 - 貓咪影像分割 (僅選取 3 種貓)
# 使用 PyTorch 與 Oxford-IIIT Pet Dataset
# https://www.robots.ox.ac.uk/~vgg/data/pets/
# 需安裝套件: torch, torchvision, numpy, matplotlib, pillow, torchinfo
# 安裝指令
# Python 3.12
# python.exe -m pip install --upgrade pip
# pip3 install --upgrade setuptools
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# pip3 install matplotlib==3.10.8
# pip3 install torchinfo==1.8.0

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchinfo import summary
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# 1. 參數設定與工具函式
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10
TARGET_CATS = [0, 1, 2]  # 選取前三種貓 (Abyssinian, Bengal, Birman)


def calculate_iou(preds, labels):
    """計算前景(Class 1)的 IoU"""
    intersection = ((preds == 1) & (labels == 1)).float().sum((1, 2))
    union = ((preds == 1) | (labels == 1)).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


# ==========================================
# 2. 資料預處理與自定義 Dataset
# ==========================================
img_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def mask_transform(mask):
    mask = mask.resize((128, 128), resample=Image.NEAREST)
    mask_np = np.array(mask)
    # Trimap: 1(前景), 3(輪廓) -> 1 (貓); 2(背景) -> 0
    binary_mask = np.where((mask_np == 1) | (mask_np == 3), 1, 0)
    return torch.from_numpy(binary_mask).long()


class CatSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, subset, mask_transform):
        self.subset = subset
        self.mask_transform = mask_transform

    def __len__(self): return len(self.subset)

    def __getitem__(self, idx):
        img, (cat_idx, mask) = self.subset[idx]
        mask = self.mask_transform(mask)
        return img, mask


# 下載與篩選資料
print("正在載入資料集...")
full_dataset = datasets.OxfordIIITPet(root='./data', split='trainval',
                                      target_types=('category', 'segmentation'),
                                      download=True, transform=img_transform)

# 過濾指定貓種索引
indices = [i for i, (_, (cat, _)) in enumerate(full_dataset) if cat in TARGET_CATS]
filtered_subset = Subset(full_dataset, indices)

# 執行 8:2 隨機劃分 (Train/Val Split)
train_size = int(0.8 * len(filtered_subset))
val_size = len(filtered_subset) - train_size
train_sub, val_sub = random_split(filtered_subset, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(42))

train_data = CatSegmentationDataset(train_sub, mask_transform)
val_data = CatSegmentationDataset(val_sub, mask_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"訓練集樣本: {len(train_data)}, 驗證集樣本: {len(val_data)}")


# ==========================================
# 3. U-Net 模型架構
# ==========================================
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


model = UNet().to(DEVICE)
print("\n=== 模型結構 ===")
summary(model, input_size=(1, 3, 128, 128))

# ==========================================
# 4. 訓練與驗證迴圈
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    # --- 訓練階段 ---
    model.train()
    train_loss, train_iou = 0, 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_iou += calculate_iou(torch.argmax(outputs, dim=1), masks)

    # --- 驗證階段 ---
    model.eval()
    val_loss, val_iou = 0, 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)
            val_loss += criterion(outputs, masks).item()
            val_iou += calculate_iou(torch.argmax(outputs, dim=1), masks)

    print(f"Epoch [{epoch + 1}/{EPOCHS}] "
          f"Train Loss: {train_loss / len(train_loader):.4f}, Train IoU: {train_iou / len(train_loader):.4f} | "
          f"Val Loss: {val_loss / len(val_loader):.4f}, Val IoU: {val_iou / len(val_loader):.4f}")

# ==========================================
# 5. 視覺化預測結果 (修正 Clipping 問題)
# ==========================================
model.eval()
with torch.no_grad():
    imgs, masks = next(iter(val_loader))
    outputs = model(imgs.to(DEVICE))
    preds = torch.argmax(outputs, dim=1).cpu()

    plt.figure(figsize=(12, 9))
    for i in range(3):
        # 反標準化以利顯示
        img_display = imgs[i].permute(1, 2, 0).numpy()
        img_display = np.clip(img_display * 0.229 + 0.485, 0, 1)

        plt.subplot(3, 3, i * 3 + 1);
        plt.imshow(img_display);
        plt.title("Image")
        plt.subplot(3, 3, i * 3 + 2);
        plt.imshow(masks[i], cmap='gray');
        plt.title("Ground Truth")

        iou_score = calculate_iou(preds[i].unsqueeze(0), masks[i].unsqueeze(0))
        plt.subplot(3, 3, i * 3 + 3);
        plt.imshow(preds[i], cmap='jet');
        plt.title(f"Pred (IoU: {iou_score:.2f})")
    plt.tight_layout();
    plt.show()