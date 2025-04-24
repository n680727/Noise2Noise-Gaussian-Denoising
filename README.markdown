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
訓練過程中，模型的 PSNR 曲線圖如下：

![image](https://github.com/user-attachments/assets/c14b5354-9480-45b1-81f3-164bcdab5cf5)


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

比較圖如下：

![image](https://github.com/user-attachments/assets/cf92bda8-c5ee-4911-8573-6fc35c9d44d2)
![image](https://github.com/user-attachments/assets/93ba5734-de1e-40aa-8abd-e0434616ce43)
![image](https://github.com/user-attachments/assets/f2d4cce3-7400-490a-94c3-4ac9e9c794aa)


這些結果顯示，Noise2Noise 模型在去雜訊效果上顯著優於傳統的中值濾波器方法，PSNR 提升了約 7.31（相較於中值濾波器）。

## 參考資料

本專案參考了 NVIDIA Research 的 [Noise2Noise 儲存庫](https://github.com/NVlabs/noise2noise)。
