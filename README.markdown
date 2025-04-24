# Noise2Noise：高斯雜訊去除

本專案實作了一個 Noise2Noise 模型，用於去除圖像中的高斯雜訊。模型使用 BSD300 資料集進行訓練，並在 Kodak24 資料集上進行驗證，驗證結果的 PSNR 為 **32.08**。本專案參考了 NVIDIA Research 的 [Noise2Noise GitHub 儲存庫](https://github.com/NVlabs/noise2noise)。

## 前置需求

- Python 3.6
- Conda 環境
- TensorFlow（支援 GPU）
- BSD300 資料集：[下載連結](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- Kodak24 資料集（用於驗證）

## 環境建置

1. **建立並啟動 Conda 環境**：
   ```bash
   conda create -n n2n python=3.6
   conda activate n2n
   ```

2. **安裝依賴套件**：
   ```bash
   conda install tensorflow-gpu
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **準備 BSD300 資料集**：
   - 從[官方網站](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)下載 BSD300 資料集。
   - 將資料集分為 `train` 和 `test` 兩個子集。
   - 使用以下指令將訓練資料集轉換為 TFRecords 格式：
     ```bash
     python dataset_tool_tf.py --input-dir datasets/BSDS300-images/BSDS300/images/train --out=datasets/bsd300.tfrecords
     ```

4. **下載 Kodak24 資料集**（用於驗證）：
   ```bash
   python download_kodak.py --output-dir=datasets/kodak
   ```

## 訓練模型

使用以下指令開始訓練 Noise2Noise 模型（針對高斯雜訊）：
```bash
python config.py train --noise=gaussian --noise2noise=true --long-train=true --train-tfrecords=datasets/bsd300.tfrecords
```

### 訓練結果
訓練過程中，模型的 PSNR 曲線圖如下（請將圖片檔案上傳至專案並更新下方路徑）：
![訓練 PSNR 曲線圖](path/to/psnr_curve.png)

## 驗證模型

使用 Kodak24 資料集驗證訓練好的模型，平均 PSNR 為 **32.08**：
```bash
python config.py validate --dataset-dir=datasets/kodak --noise=gaussian --network-snapshot=<path_to_network_final.pickle>
```
*注意*：請將 `<path_to_network_final.pickle>` 替換為實際的模型權重檔案路徑。

## 圖像去雜訊

訓練完成後，可使用以下指令對單張圖像進行去雜訊：
```bash
python config.py infer-image --image=C:\Users\5ji6r\noise2noise\img\man_log.png --out=C:\Users\5ji6r\noise2noise\img\man_log_denoise.png --network-snapshot=C:\Users\5ji6r\noise2noise\network_final-gaussian-n2n.pickle
```
- `--image`：輸入含雜訊的圖像路徑
- `--out`：輸出去雜訊後的圖像路徑
- `--network-snapshot`：指定訓練好的模型權重檔案

### 去雜訊效果比較
以下是使用不同方法處理含高斯雜訊圖像的 PSNR 比較，顯示 Noise2Noise 模型相較於傳統中值濾波器的優越性能：
- 含高斯雜訊圖像：**28.43**
- 中值濾波器去雜訊：**31.03**
- Noise2Noise 模型去雜訊：**38.34**

比較圖如下（請將圖片檔案上傳至專案並更新下方路徑）：
![含高斯雜訊圖像](path/to/noisy_image.png)
![中值濾波器去雜訊](path/to/median_filtered_image.png)
![Noise2Noise 去雜訊](path/to/denoised_image.png)

這些結果顯示，Noise2Noise 模型在去雜訊效果上顯著優於傳統的中值濾波器方法，PSNR 提升了約 7.31（相較於中值濾波器）。

## 參考資料

本專案參考了 NVIDIA Research 的 [Noise2Noise 儲存庫](https://github.com/NVlabs/noise2noise)。感謝他們的開源貢獻！

## 聯繫

如有問題或建議，歡迎提交 Issue 或 Pull Request！