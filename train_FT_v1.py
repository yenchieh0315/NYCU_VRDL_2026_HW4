import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image
import math
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import csv

# ==========================================
# 工具函式 (Utils)
# ==========================================
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse.item()))

# ==========================================
# 1. 資料集與前處理 (Dataset & Augmentation)
# ==========================================
class RestorationDataset(Dataset):
    def __init__(self, degraded_dir, clean_dir, patch_size=256):
        super().__init__()
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.patch_size = patch_size
        
        # 取得檔名 (確保 degraded 和 clean 檔名能對應，請依據作業實際命名邏輯修改)
        # 作業提示：rain-1.png 對應 rain_clean-1.png
        self.degraded_files = [f for f in os.listdir(degraded_dir) if f.endswith(('.png', '.jpg'))]
        
    def __len__(self):
        return len(self.degraded_files)

    def __getitem__(self, idx):
        deg_name = self.degraded_files[idx]
        
        # 檔名對映邏輯 (依據作業講義 p.11)
        if 'rain' in deg_name:
            clean_name = deg_name.replace('rain-', 'rain_clean-')
        else:
            clean_name = deg_name.replace('snow-', 'snow_clean-')

        deg_path = os.path.join(self.degraded_dir, deg_name)
        clean_path = os.path.join(self.clean_dir, clean_name)
        
        img_deg = Image.open(deg_path).convert('RGB')
        img_clean = Image.open(clean_path).convert('RGB')
        
        # 轉為 Tensor
        img_deg = TF.to_tensor(img_deg)
        img_clean = TF.to_tensor(img_clean)
        
        # (1) Random Crop 隨機裁切 (256x256)
        i, j, h, w = self._get_crop_params(img_deg, self.patch_size)
        img_deg = TF.crop(img_deg, i, j, h, w)
        img_clean = TF.crop(img_clean, i, j, h, w)
        
        # (2) Random Horizontal Flip 隨機水平翻轉 (機率 0.5)
        if random.random() > 0.5:
            img_deg = TF.hflip(img_deg)
            img_clean = TF.hflip(img_clean)
            
        return img_deg, img_clean

    def _get_crop_params(self, img, output_size):
        c, h, w = img.shape
        th, tw = output_size, output_size
        if h == th and w == tw:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw


# ==========================================
# 2. 網路模組建構 (Model Components)
# ==========================================

# -- NAFNet Block (極輕量卷積特徵提取) --
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = nn.LayerNorm(c)
        self.conv1 = nn.Conv2d(c, c * 2, 1)
        self.dwconv = nn.Conv2d(c * 2, c * 2, 3, 1, 1, groups=c * 2)
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(c, c, 1)
        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        
        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Conv2d(c, c * 2, 1), SimpleGate(), nn.Conv2d(c, c, 1))

    def forward(self, x):
        identity = x
        # NAF Forward
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv1(x_norm)
        x = self.dwconv(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv2(x) + identity
        
        # FFN Forward
        identity = x
        x_norm = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.ffn(x_norm) + identity
        return x

# -- Transformer Block (用於底層 Bottleneck) --
class TransformerBlock(nn.Module):
    def __init__(self, c, num_heads=4):
        super().__init__()
        # 這裡為了簡化並節省 VRAM，使用基礎的通道自注意力 (MDTA-style)
        self.norm1 = nn.LayerNorm(c)
        self.qkv = nn.Conv2d(c, c*3, 1)
        self.proj = nn.Conv2d(c, c, 1)
        self.num_heads = num_heads
        
        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Conv2d(c, c*2, 1), nn.GELU(), nn.Conv2d(c*2, c, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        identity = x
        
        # Channel-wise Self Attention
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).view(b, c, h, w)
        
        x = self.proj(out) + identity
        
        # FFN
        identity = x
        x_norm = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.ffn(x_norm) + identity
        return x

# -- Coordinate Attention (空間注意力) --
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = F.relu(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv2(x_h).sigmoid()
        a_w = self.conv3(x_w).sigmoid()
        out = identity * a_h * a_w
        return out

# -- SK-Net Bridge (多尺度特徵融合 Skip Connection) --
class SKBridge(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, groups=features)
        self.conv2 = nn.Conv2d(features, features, 5, padding=2, groups=features)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(features, features // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, features * 2, 1)
        )

    def forward(self, x):
        U1 = self.conv1(x)
        U2 = self.conv2(x)
        U = U1 + U2
        
        Z = self.fc(U)
        a, b = torch.split(Z, U1.size(1), dim=1)
        a = a.unsqueeze(dim=1)
        b = b.unsqueeze(dim=1)
        A = torch.cat([a, b], dim=1).softmax(dim=1)
        
        V = U1 * A[:, 0] + U2 * A[:, 1]
        return V

# ==========================================
# 3. 核心 PromptIR Hybrid 模型架構
# ==========================================
class PromptIR_Hybrid(nn.Module):
    def __init__(self, in_c=3, out_c=3, dim=32):
        super().__init__()
        self.embed = nn.Conv2d(in_c, dim, 3, 1, 1)
        
        # Encoder (NAFBlocks)
        self.enc1 = NAFBlock(dim)
        self.down1 = nn.Conv2d(dim, dim*2, 2, 2)
        self.enc2 = NAFBlock(dim*2)
        self.down2 = nn.Conv2d(dim*2, dim*4, 2, 2)
        
        # SK-Net Bridges (用在 Skip Connections)
        self.sk_bridge1 = SKBridge(dim)
        self.sk_bridge2 = SKBridge(dim*2)
        
        # Bottleneck (2 Transformer Blocks - 你要求的特殊設計)
        self.bottleneck1 = TransformerBlock(dim*4)
        self.bottleneck2 = TransformerBlock(dim*4)
        self.bottleneck3 = TransformerBlock(dim*4)
        self.bottleneck4 = TransformerBlock(dim*4)
        
        # Decoder (NAFBlocks)
        self.up2 = nn.ConvTranspose2d(dim*4, dim*2, 2, 2)
        self.dec2 = NAFBlock(dim*2)
        self.up1 = nn.ConvTranspose2d(dim*2, dim, 2, 2)
        self.mid_transformer1 = TransformerBlock(dim)
        self.mid_transformer2 = TransformerBlock(dim)
        
        self.dec1 = NAFBlock(dim)
        
        # Spatial Attention (Coordinate Attention)
        self.coord_att = CoordAtt(dim)
        
        # Output
        self.mapping = nn.Conv2d(dim, out_c, 3, 1, 1)

        # Prompt Generation (極簡版示範)
        self.prompt_embed = nn.Parameter(torch.randn(1, dim*4, 1, 1))

    def forward(self, x):
        feat = self.embed(x)
        
        # Encode
        e1 = self.enc1(feat)
        e2 = self.enc2(self.down1(e1))
        b = self.down2(e2)
        
        # Bottleneck (Transformer) + Prompt Interaction
        b = self.bottleneck1(b)
        # 將 Prompt 與底層特徵進行交互 (簡單的通道相乘/加)
        b = b * torch.sigmoid(self.prompt_embed) 
        b = self.bottleneck2(b)
        b = self.bottleneck3(b)
        b = self.bottleneck4(b)
        
        # Decode & SK-Bridge Skip Connections
        d2 = self.dec2(self.up2(b) + self.sk_bridge2(e2))
        d1_up = self.up1(d2) + self.sk_bridge1(e1)
        d1_up = self.mid_transformer1(d1_up)
        d1_up = self.mid_transformer2(d1_up)
        d1 = self.dec1(d1_up)
        
        # Spatial Attention Enhancement
        d1 = self.coord_att(d1)
        
        out = self.mapping(d1) + x # Global Residual
        return out


# ==========================================
# 4. 損失函數 (Loss Functions)
# ==========================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss

class FFTLoss(nn.Module):
    def forward(self, x, y):
        # 轉換至頻域進行 L1 Loss 比較 (有效抑制高頻的雨雪雜訊)
        fft_x = torch.fft.rfft2(x)
        fft_y = torch.fft.rfft2(y)
        loss = torch.mean(torch.abs(fft_x - fft_y))
        return loss


# ==========================================
# 5. 主訓練迴圈 (Training Loop)
# ==========================================
def main():
    # --- 參數設定 ---
    TRAIN_DEGRADED = './hw4_release_dataset/train/degraded'
    TRAIN_CLEAN = './hw4_release_dataset/train/clean'
    BATCH_SIZE = 16     # 8GB VRAM 的安全值
    EPOCHS = 50
    LR = 5e-5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device} (RTX 4060 8GB Ready)')

    # --- 資料加載 ---
    train_dataset = RestorationDataset(TRAIN_DEGRADED, TRAIN_CLEAN, patch_size=256)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    # --- 模型、損失函數、優化器 ---
    model = PromptIR_Hybrid(dim=32).to(device)
    
    previous_weight = './V3/v3_pth/v3_epoch200.pth'
    model.load_state_dict(torch.load(previous_weight, map_location=device))
    print(f"成功載入預訓練權重：{previous_weight}，開始漸進式微調！")
    
    criterion_charbonnier = CharbonnierLoss()
    criterion_fft = FFTLoss()
    
    # 使用 AdamW
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # T_max 改吃這一次設定的 50 Epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # 混合精度訓練 GradScaler (防 OOM 加速神器)
    #scaler = GradScaler()

    log_filename = 'training_log_v3_fine-tuning.csv'
    csv_file = open(log_filename, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Average Loss', 'Average PSNR', 'Learning Rate']) # 寫入標題列
    
    # --- 開始訓練 ---
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0 # 新增：用來記錄這一個 Epoch 累積的 PSNR
        
        pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{EPOCHS}]')
        for degraded_img, clean_img in pbar:
            degraded_img = degraded_img.to(device)
            clean_img = clean_img.to(device)
            
            optimizer.zero_grad()
            
            # 使用混合精度 (AMP) 節省 8GB VRAM
            with torch.autocast('cuda', dtype=torch.bfloat16):
                restored_img = model(degraded_img)
                
                # 總 Loss = Charbonnier Loss + 0.05 * FFT Loss (比重需自己微調)
                loss_charb = criterion_charbonnier(restored_img, clean_img)
                loss_fft = criterion_fft(restored_img, clean_img)
                loss = loss_charb + 0.05 * loss_fft

            # 反向傳播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            # 取得目前的 Loss 並計算 PSNR
            current_loss = loss.item()
            with torch.no_grad(): # 計算 PSNR 不需要梯度
                current_psnr = calculate_psnr(restored_img, clean_img)
            epoch_psnr += current_psnr
            pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'PSNR': f'{current_psnr:.2f}'})
            
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = epoch_loss / len(train_loader)
        avg_psnr = epoch_psnr / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Avg Loss: {avg_loss:.4f}, Avg PSNR: {avg_psnr:.2f}, LR: {current_lr:.6f}")

        csv_writer.writerow([epoch+1, avg_loss, avg_psnr, current_lr])
        csv_file.flush() # 確保每一輪結束都立刻寫入硬碟，避免意外中斷資料遺失
        
        # 定期存檔
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'v3_epoch{epoch+1}_fine-tuning.pth')
    csv_file.close() # 訓練結束，關閉檔案
    print("Training Complete! Model saved.")

if __name__ == '__main__':
    main()