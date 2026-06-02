import os
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm
import zipfile

# 從你的 train.py 匯入你設計好的模型架構
from train import PromptIR_Hybrid 

def pad_image(img_tensor, factor=4):
    """
    因為我們的 U-Net 架構有 2 次 downsample (2x2=4)，
    輸入圖片的長寬必須是 4 的倍數，否則 Skip-connection 尺寸會對不上。
    這個函式會自動幫圖片補邊 (Padding)。
    """
    _, h, w = img_tensor.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    # 使用 reflection padding 邊界反射填充，對修復任務較自然
    img_padded = F.pad(img_tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect')
    return img_padded, h, w

def main():
    # --- 1. 設定路徑與參數 ---
    # 請確認這是作業測試集的資料夾路徑
    test_dir = './hw4_release_dataset/test/degraded'  
    # 填入你訓練好的權重檔名 (最後一個 epoch)
    model_weight_path = './V3/v3_pth/v3_epoch200.pth' 
    output_npz = 'pred.npz'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # --- 2. 載入模型 ---
    model = PromptIR_Hybrid(dim=32).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval() # 設定為推論模式
    print("模型權重載入成功！")

    # --- 3. 開始推論 ---
    images_dict = {} 
    test_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))]
    
    with torch.no_grad(): # 推論時不計算梯度，節省 VRAM
        for filename in tqdm(test_files, desc="Processing Test Images with TTA"):
            img_path = os.path.join(test_dir, filename)
            
            # 讀取圖片並轉為 Tensor (C, H, W)，數值 0~1
            img = Image.open(img_path).convert('RGB')
            img_tensor = TF.to_tensor(img).to(device)
            
            # 補邊到 4 的倍數
            img_padded, original_h, original_w = pad_image(img_tensor, factor=4)
            
            with torch.autocast('cuda', dtype=torch.bfloat16):
                # ==================================================
                # 加入 Test-Time Augmentation (TTA) - 水平翻轉
                # ==================================================
                
                # 1. 原始圖片推論 (Pass 1)
                out_original = model(img_padded)
                
                # 2. 建立水平翻轉圖片 (Pass 2)
                # 維度是 [B, C, H, W]，水平翻轉是針對 W (即第 3 個維度)
                img_flipped = torch.flip(img_padded, dims=[3]) 
                out_flipped = model(img_flipped)
                
                # 將翻轉圖片的預測結果「翻轉回來」
                out_flipped_reversed = torch.flip(out_flipped, dims=[3])
                
                # 3. 將兩次結果進行平均 (Ensemble)
                restored_padded = (out_original + out_flipped_reversed) / 2.0
                
                # ==================================================

            # 裁切回原本的圖片大小 (去除剛剛補的邊)
            restored = restored_padded[0, :, :original_h, :original_w]
            
            # 將數值限制在 0~1 之間 (防止模型輸出異常亮點)
            restored = torch.clamp(restored, 0.0, 1.0)
            
            # --- 4. 轉換為 CodaBench 規定的格式 ---
            restored_np = (restored.cpu().numpy() * 255.0).round().astype(np.uint8)
            assert restored_np.shape == (3, original_h, original_w), f"Shape 錯誤: {restored_np.shape}"
            images_dict[filename] = restored_np

    # --- 5. 儲存為 pred.npz 並自動打包 ZIP ---
    print(f"\n正在儲存 {len(images_dict)} 張圖片至 {output_npz} ...")
    np.savez(output_npz, **images_dict)
    
    zip_filename = 'submission.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_npz)
        
    print(f"完成！已自動打包為 {zip_filename}，請將這個檔案上傳至 CodaBench！")

if __name__ == '__main__':
    main()