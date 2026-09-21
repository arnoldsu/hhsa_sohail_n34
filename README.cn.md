# 基於 HHSA 雙分支神經網路的 Niño3.4 事件時序與強度預報

## Conceptual framework

The experiment applies Sohail's neural-network approach to HHSA-derived
Niño3.4 event and strength components.

![Application of Sohail's NN with HHSA](figures/sohail_nn_hhsa_framework.png)

**Figure 1. Application of Sohail's NN with HHSA.**  
Sohail's neural-network framework is adapted to HHSA by separating
event/timing information (IMF, IF and IP) from strength/amplitude
information (IA and AM-IMFs). Separate neural networks are then used
to predict the future event phase and amplitude before reconstructing
the future Niño3.4 state.

## 摘要

本研究提出一套可解釋的 Niño3.4 預報架構，目的不是讓單一神經網路直接由歷史 Niño3.4 猜測未來指數，而是先利用 Hilbert–Huang spectral analysis（HHSA）把振盪的「時序」與「強度」分開，再分別建模。第一層 masking empirical mode decomposition（EMD）將 Niño3.4 分解為六個振盪 intrinsic mode functions（IMFs）及一個非振盪殘差。每個 IMF 以 direct-quadrature 方法表示為瞬時振幅 (A_i(t)) 與瞬時相位 (phi_i(t))。第二層 EMD 再將 (A_i(t)) 分解為不同調制尺度的 amplitude-modulation IMFs（AM-IMFs）。Event neural network 由 IMF、instantaneous frequency 與相位 phasor 預測未來相位；Strength neural networks 則分別預測每一個 AM-IMF。最終透過

$$
\widehat{IMF}_i(t+k)=\widehat A_i(t+k)\cos\widehat\phi_i(t+k)
$$

及

$$
\widehat x(t+k)=\sum_i\widehat{IMF}_i(t+k)+\widehat r(t+k)
$$

完成物理重建。研究比較 persistence、direct neural network、single-state HHSA residual network、雙分支物理重建與 fusion residual network。Full-record offline diagnostic 中，雙分支物理重建在 3、6、9、12 個月的 (R^2) 分別為 0.874、0.792、0.692、0.619，優於 direct NN 與 single-state HHSA network。然而，在對 2015、2018、2020、2023 El Niño 峰值進行 causal-prefix hindcast 時，雙分支方法未能勝過 persistence。結果顯示，顯式分離 event timing 與 amplitude strength 在離線表示學習上有價值，但 EMD endpoint effects、跨 forecast-origin 的 mode identity 不穩定，以及強事件振幅低估，仍阻礙其成為可靠的即時預報方法。

**關鍵詞：** Niño3.4、ENSO、HHSA、Hilbert–Huang transform、EMD、instantaneous amplitude、instantaneous phase、AM-IMF、neural network、causal hindcast

---

## 1. 研究背景與問題

Niño3.4 指數同時包含至少兩類資訊：

1. 事件何時開始、何時到達峰值，以及振盪處於哪一個相位；
2. 事件最終能發展多強。

一般 direct neural network 將這些資訊混合在同一個目標中：

$$
[x(t-L+1),\ldots,x(t)]\rightarrow\widehat x(t+k).
$$

若預報失敗，這種模型很難回答錯誤究竟來自 phase/timing，還是 amplitude/strength。本研究的核心問題因此是：

> 顯式分離 ENSO 事件時序與事件強度，是否能提高 Niño3.4 預報，並讓失敗原因更容易解釋？

概念架構為：

$$
\boxed{
x(t)\xrightarrow{HHSA}
\begin{cases}
IMF_i,IF_i,\phi_i &\rightarrow NN_{event}\rightarrow\widehat\phi_i,\\
A_i,AMIMF_{ij} &\rightarrow NN_{strength}\rightarrow\widehat A_i.
\end{cases}}
$$

然後利用 HHSA 的解析形式重建未來訊號，而不是再交由一個無物理限制的黑箱直接產生最終答案。

---

## 2. Sohail 方法的來源與本研究的改造

### 2.1 原始 Sohail 工作

Sohail、Zika 與 Ehmen 的研究 *How accurate are salinity measurements around Antarctica? A machine learning based approach* 使用機器學習評估南極周邊鹽度觀測品質。正式論文 DOI 為 [10.1088/3049-4753/ae7113](https://doi.org/10.1088/3049-4753/ae7113)。作者提供的程式、資料與製圖資源收錄於 Zenodo，記錄 DOI 為 [10.5281/zenodo.14010532](https://doi.org/10.5281/zenodo.14010532) [1,2]。

其可重現程式中的方法特徵包括：

- 將描述環境或觀測狀態的多個變數組成 state vector；
- 將月份轉換為 sine/cosine 週期座標，避免 12 月與 1 月在數值上不連續；
- 使用 feed-forward neural network（FFN）；
- 使用帶 skip connections 的 residual FFN；
- residual 架構先將輸入映射至 64 維，接著使用三個 `256 → 128 → 64` dense blocks，每個 block 的輸出與 64 維 residual 相加；
- 使用 Adam optimizer 與 mean-squared-error loss；
- 使用 early stopping、learning-rate reduction 與 checkpointing；
- 針對不同資料分割方式評估 out-of-sample 表現。

### 2.2 不能直接照搬的部分

Sohail 原研究不是 ENSO 時間序列預報，也沒有使用 EMD、HHSA、instantaneous amplitude 或 instantaneous phase。因此本研究中的「Sohail-style」只指以下可轉移原則：

1. 以具有科學意義的 state variables 描述系統狀態；
2. 使用 sine/cosine 表示週期座標；
3. 比較普通 FFN 與 residual FFN；
4. 以 out-of-sample 評估檢驗附加 state information 是否有價值。

本研究沒有宣稱 Sohail 等人提出過 HHSA–ENSO 模型，也沒有直接複製其大型網路。Niño3.4 只有 943 個月，樣本量遠小於其海洋剖面資料，因此本研究縮小 hidden dimension，採 chronological splitting，並將 residual-learning 原則嵌入時間預報問題。

### 2.3 本研究的主要改造

Sohail-style state vector 在本研究中改寫為：

$$
S_E(t)=
[IMF_i(t),IF_i(t),\cos\phi_i(t),\sin\phi_i(t)]_{i=1}^{6}
$$

供 Event NN 使用；Strength branch 則不把全部尺度塞進單一 state network，而是讓每個 AM-IMF 擁有獨立的小型 NN。這項改造是本研究自己的方法貢獻，而不是 Sohail 原模型的一部分。

---

## 3. 資料

### 3.1 Niño3.4 時序

本研究使用工作目錄既有的月平均 Niño3.4 anomaly：

```text
data/nino34_monthly.csv
```

資料特徵如下：

- 起始：1948-01；
- 結束：2026-07；
- 樣本數：943 個月；
- 單位：°C；
- 缺失值：0；
- sampling rate：12 samples year(^{-1})。

沒有下載或混入另一套 Niño3.4 資料，也沒有在 EMD 前平滑原始指數。

### 3.2 時間切分

Offline experiment 使用 chronological split：

```text
training      earliest 70%
validation    next 15%
test          final 15%
```

不存在 random time split。所有 predictor 與 target scaling parameters 只由 training period 估計。

---

## 4. HHSA 分解

### 4.1 第一層 masking EMD

令原始 Niño3.4 指數為 (x(t))。第一層分解為：

$$
x(t)=\sum_{i=1}^{6}IMF_i(t)+r(t).
$$

實際輸出包含六個 oscillatory IMFs 和一個 residual/trend。使用的是專案既有、由 MATLAB masking-EMD 程式移植的 Python 實作，而不是第三方 `emd` 套件。

本研究設定 `upsample_level=0`。原因是現有 Python port 在 `upsample_level=1` 的降採樣步驟會移除第一個高頻 mode，造成第一層無法精確重建。Level 0 保留 masking-EMD 核心演算法，且滿足 reconstruction sanity check。

### 4.2 Direct-quadrature instantaneous quantities

每個 oscillatory IMF 表示為：

$$
IMF_i(t)=A_i(t)\cos\phi_i(t),
$$

其中：

- (A_i(t))：instantaneous amplitude（IA），描述 strength；
- (phi_i(t))：instantaneous phase，描述 event/oscillation position；
- (f_i(t))：instantaneous frequency，描述 phase 的演化速率。

$$
f_i(t)=\frac{1}{2\pi}\frac{d\phi_i(t)}{dt}.
$$

本研究沿用專案的 direct-quadrature 與 PCHIP normalization，而非一般 FFT band-pass 或任意 Hilbert package。

### 4.3 第二層 amplitude decomposition

每個第一層 IA 再接受第二次 masking EMD：

$$
A_i(t)=\sum_j AMIMF_{ij}(t)+r_i^{AM}(t).
$$

這個層級用來分離快速與緩慢的 amplitude modulation。所有有效 AM-IMFs 及其 amplitude residual 必須先通過：

$$
A_i(t)\approx\sum_j AMIMF_{ij}(t)+r_i^{AM}(t).
$$

失敗或不存在的 component 不會以 NaN 輸入網路；只有確實具有非零 training variance 的 component 會建立 Strength NN。

---

## 5. Dual-NN 架構

## 5.1 Event NN：預測何時發生

Event branch 的輸入是最近 60 個月的：

$$
X_E(t)=
\{IMF_i,IF_i,\cos\phi_i,\sin\phi_i\}_{t-59:t}.
$$

不用裸相位 (phi) 作 target，因為 (phi) 在 (2\pi) 處存在不連續。網路輸出 future phasor：

$$
NN_E(X_E)ightarrow
(\widehat{\cos\phi_i(t+k)},\widehat{\sin\phi_i(t+k)}).
$$

預測後將 phasor 投影回 unit circle：

$$
(\hat c,\hat s)leftarrow
\frac{(\hat c,\hat s)}{\sqrt{\hat c^2+\hat s^2}}.
$$

這同時避免振幅誤差污染 phase 表示。

Event branch 的主要指標是 circular phase error：

$$
e_{\phi}=
\cos^{-1}
\left[
\widehat{\boldsymbol p}\cdot\boldsymbol p
\right],
$$

其中 (oldsymbol p=(\cos\phi,\sin\phi))。另報告 mean phasor cosine。

## 5.2 Strength NNs：預測會多強

每一個有效 (AMIMF_{ij}) 使用自己的小型網路：

$$
AMIMF_{ij}(t-L_{ij}+1:t)
\rightarrow
\widehat{AMIMF}_{ij}(t+k).
$$

history length 不固定為同一值，而是根據 training-period component median timescale (T_{ij})：

$$
L_{ij}=\operatorname{clip}(2T_{ij},12,60).
$$

這使快速 modulation 使用較短窗口、緩慢 modulation 使用較長窗口。預測 IA 為：

$$
\widehat A_i(t+k)=
\sum_j\widehat{AMIMF}_{ij}(t+k)
+\widehat r_i^{AM}(t+k).
$$

individual AM-IMFs 可以為正或負；不能對每一項套用 exponential 或強迫為正。物理上必須非負的是重建後的 (A_i)，而非每個 modulation component。

## 5.3 物理重建

Event 與 Strength branches 完成後，不需要第三個黑箱模型即可重建：

$$
\widehat{IMF}_i(t+k)
=
\widehat A_i(t+k)
\widehat{\cos\phi_i(t+k)}.
$$

最後：

$$
\boxed{
\widehat x(t+k)
=
\sum_{i=1}^{6}
\widehat A_i(t+k)
\widehat{\cos\phi_i(t+k)}
+\widehat r(t+k).
}
$$

此方法稱為 M4 Dual physical reconstruction。

---

## 6. 比較模型 M0–M5

### M0：Persistence

$$
\widehat x(t+k)=x(t).
$$

任何複雜方法都必須先勝過這個基準。

### M1：Direct Niño3.4 NN

使用最近 60 個月原始 Niño3.4，直接預測 1、3、6、9、12 月目標。這是沒有 HHSA 的 neural-network baseline。

### M2：HHSA single-state residual network

輸入單一 forecast-origin 的完整 HHSA state：原始指數、IMFs、IA、IF、phase phasors、second-level AM-IMFs 和 residual。其目的是測試「HHSA state variables 是否包含附加資訊」，但它仍直接輸出未來 Niño3.4，沒有強制遵守 (A\cos\phi)。

本研究採用縮小的 residual network。其設計概念來自 Sohail 原碼的 residual dense blocks，但 hidden dimension 縮小，以配合只有數百個 training samples 的月資料。

### M3：Event 與 Strength branches

M3 不一定產生單一 final Niño3.4 score；它分別回答：

- phase/timing 是否可預測？
- IA 與 individual AM-IMFs 是否可預測？
- 哪一個 branch 最先隨 lead 崩潰？

### M4：Dual physical reconstruction

M4 將 M3 的 branch outputs 透過 (A\cos\phi) 合成，是本研究的主要模型。

### M5：Fusion residual network

M5 把預測的 amplitude、phase phasor 與 residual 交給一個小型 residual network，測試 data-driven fusion 是否能修正物理合成的系統誤差。為減少 stacking leakage，M5 只由 validation-period base predictions 訓練，不使用 base-model training predictions。

---

## 7. 訓練與評估

### 7.1 神經網路訓練

本研究使用 PyTorch，主要設定包括：

- random seed：42；
- hidden dimension：48；
- optimizer：AdamW；
- learning rate：(2\times10^{-3})；
- weight decay：(2\times10^{-4})；
- batch size：32；
- maximum epochs：80；
- early-stopping patience：10；
- CPU threads：1。

所有 feature-wise mean/std 與 target mean/std 只由 training subset 計算。

### 7.2 訊號預報指標

對每一個 lead 計算：

$$
RMSE=\sqrt{\frac1N\sum_n(\hat x_n-x_n)^2},
$$

$$
MAE=\frac1N\sum_n|\hat x_n-x_n|,
$$

$$
R^2=1-\frac{\sum_n(\hat x_n-x_n)^2}{\sum_n(x_n-\bar x)^2}.
$$

另計算 correlation、regression slope 與 variance ratio：

$$
VR=\frac{\operatorname{var}(\hat x)}{\operatorname{var}(x)}.
$$

若 correlation 尚可但 slope 或 variance ratio 明顯小於 1，代表模型可能抓到 timing，卻把事件 amplitude 壓縮。

### 7.3 Branch-specific metrics

Event branch 報告 circular MAE 與 phasor cosine。Strength branch 對每一個 IA 和 AM-IMF 報告 correlation、RMSE、MAE、(R^2)。這讓研究能分辨最終誤差主要來自 phase 還是 amplitude。

---

## 8. Offline diagnostic 結果

### 8.1 完整 Niño3.4 skill

| Lead（月） | M0 Persistence (R^2) | M1 Direct NN | M2 Single HHSA | M4 Dual physical | M5 Fusion |
|---:|---:|---:|---:|---:|---:|
| 1 | **0.920** | 0.727 | 0.736 | 0.893 | -0.013 |
| 3 | 0.603 | 0.356 | 0.719 | **0.874** | -1.389 |
| 6 | -0.077 | -0.090 | 0.713 | **0.792** | -0.171 |
| 9 | -0.677 | -0.272 | 0.628 | **0.692** | -0.220 |
| 12 | -1.038 | -0.327 | 0.509 | **0.619** | -1.155 |

主要發現：

1. 一個月預報仍由 persistence 最佳；
2. 3–12 個月時，M4 明顯優於 persistence、direct NN 與 M2；
3. M2 顯示 HHSA state variables 確實含有比 raw lags 更容易被網路利用的 offline structure；
4. M4 又優於 M2，支持 timing/strength 分離與物理合成的表示方法；
5. M5 fusion 嚴重失敗，顯示小 validation sample 上的額外 fusion network 容易 overfit，不能因架構更複雜就假定更好。

### 8.2 Event branch

| Lead（月） | Circular MAE（rad） | Mean phasor cosine |
|---:|---:|---:|
| 1 | 0.545 | 0.760 |
| 3 | 0.639 | 0.687 |
| 6 | 0.725 | 0.626 |
| 9 | 0.848 | 0.539 |
| 12 | 0.846 | 0.542 |

Event timing skill 隨 lead 逐步下降，但到 9–12 個月仍保留部分 phase alignment。這與「timing 可能比 amplitude 更有長期結構」的假設一致，但不能單憑 offline decomposition 作 operational 結論。

---

## 9. Causal-prefix El Niño hindcast

### 9.1 實驗設計

為避免把 full-record modes 當成即時 predictor，本研究另對最後 15% 中的四個 El Niño peaks 進行 forecast-like hindcast：

- 2015-11；
- 2018-11；
- 2020-01；
- 2023-11。

對每一事件與每一個 3、6、9 月 lead：

1. 在 peak 前對應 forecast origin 截斷資料；
2. 只在該 prefix 上重新進行第一、第二層 HHSA；
3. 重新訓練 Event NN 與所有 Strength NNs；
4. 預測峰值月份；
5. 與 persistence、direct NN、single HHSA network 比較。

### 9.2 結果

| Lead（月） | Persistence RMSE | Direct NN | Single HHSA | Dual physical |
|---:|---:|---:|---:|---:|
| 3 | **0.632** | 1.413 | 0.870 | 1.191 |
| 6 | **1.011** | 1.531 | 1.430 | 1.303 |
| 9 | 1.617 | **1.477** | 1.901 | 1.794 |

M4 未能勝過 persistence。特別是強事件存在明顯低估。這表示 offline M4 的高 (R^2) 不能直接解釋為即時 event forecast skill。

可能原因包括：

1. EMD endpoint effects：forecast origin 位於 decomposition 邊界；
2. mode identity instability：不同 prefix 的同一 IMF 編號不一定代表同一尺度；
3. 強 ENSO 的快速非線性增長在歷史樣本中稀少；
4. 第二層慢 amplitude modes 在 full record 上非常平滑，造成 offline predictability 偏高；
5. event sample 僅四個，無法穩定估計 correlation 或 neural-network generalization。

---

## 10. 2026-08 至 2028-12 實驗性預報

本專案使用 M4 架構另訓練 29 個月 multi-output models，輸出：

- 每個 IMF 的 forecast IA；
- 每個 IMF 的 forecast phase cosine/carrier；
- 每個 forecast IMF；
- forecast residual；
- 合成 Niño3.4。

完整結果為：

```text
results/dual_hhsa_forecast_to_2028_12.csv
```

2028-12 的點預報為約 1.739 °C。然而這個數字不應視為 operational outlook，原因是：

- 模型訓練使用 retrospective full-record HHSA features；
- 29 個月顯著超出 causal event hindcast 顯示的可靠範圍；
- 沒有經過 ensemble calibration 或 probabilistic verification；
- causal peak experiment 未勝過 persistence；
- 長 lead phase errors 可因 (A\cos\phi) 的非線性而產生大幅訊號誤差。

因此此檔案的主要用途是檢查 component-level behavior 與形成可驗證的未來假設，而不是提供決策型氣候預報。

---

## 11. 資料洩漏與可重現性審核

### 11.1 Offline experiment

Offline M0–M5 在完整 record 上先做 HHSA，再切 train/test。因此 decomposition 本身可能把 test-period boundary information 帶入 historical modes。這些結果只能標示為：

```text
OFFLINE / DIAGNOSTIC
NOT STRICTLY REAL-TIME
```

### 11.2 已控制的部分

- chronological split；
- 沒有 random time shuffle；
- normalization 只 fit training subset；
- validation 用於 early stopping；
- test 不參與 model fitting；
- M5 只用 validation-period base predictions 訓練；
- causal events 的 predictor decomposition 截止於 forecast origin；
- 原始輸入 NaN 數為零。

### 11.3 尚未完全解決的問題

即使 causal event predictor 沒有使用 event future，prefix 內的 historical training samples 仍共享一次「截至當前 origin」的 decomposition。最嚴格的 pseudo-operational evaluation 應對每一個 historical training origin 也重新分解一次，但計算量非常大。未來研究應建立 rolling decomposition cache，並用 frequency-based mode matching 追蹤跨 origin 的 carrier identity。

---

## 12. 如何重現

進入新專案：

```bash
cd /g/data/p66/ars599/HHSA_WK/hhsa_n34/hhsa_sohail_n34
```

限制 BLAS/PyTorch 為單 CPU thread：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

依序執行：

```bash
python src/run_dual_hhsa.py
python src/run_causal_events.py
python src/forecast_future.py
python src/finalize_report.py
```

主要設定位於：

```text
config.json
```

### 12.1 主程式

- `src/run_dual_hhsa.py`：M0–M5 offline experiment、branch metrics、models、figures；
- `src/run_causal_events.py`：held-out El Niño causal-prefix hindcasts；
- `src/forecast_future.py`：M4 future component forecast；
- `src/finalize_report.py`：branch figures 與 README。

### 12.2 主要輸出

- `results/metrics.csv`：M0–M5 與 individual AM-IMF metrics；
- `results/predictions.csv`：逐 target prediction；
- `results/branch_metrics.csv`：event/strength branch skill；
- `results/strength_component_histories.csv`：每個 AM-IMF 的 history length；
- `results/causal_event_predictions.csv`：逐事件逐 lead 預測；
- `results/causal_event_metrics.csv`：causal event summary；
- `results/dual_hhsa_forecast_to_2028_12.csv`：未來 component forecasts；
- `results/leakage_audit.json`：資料洩漏審核；
- `results/resource_usage.csv`：CPU/RAM time series；
- `models/`：PyTorch model weights；
- `figures/`：skill、time-series、event、future 與 branch figures。

---

## 13. 資源使用

Offline M0–M5：

- internal wall time：約 31 秒；
- `/usr/bin/time` wall time：約 39 秒；
- 平均 CPU：約 95%；
- peak Python RSS：約 470 MiB。

Causal event hindcast：

- wall time：約 6 分 55 秒；
- CPU：約單核心 96%；
- peak RSS：約 436 MiB。

Future M4 forecast：

- wall time：約 53 秒；
- peak RSS：約 443 MiB。

---

## 14. 討論

本研究最重要的正面結果不是「HHSA 已經可以可靠預測 ENSO」，而是證明了一套可被分解、診斷與否證的架構。M4 在 offline test 明顯優於 direct NN 與 single-state HHSA，表示 event/strength factorization 可能比將所有 features 交給單一網路更容易學習。Event phase skill 與 Strength component skill 可以被分別追蹤，因此能回答普通 direct LSTM 無法回答的問題。

然而，causal event test 否定了過度樂觀的解讀。當 EMD 只能看到 forecast origin 以前的資料時，carrier 與 amplitude 的邊界估計改變，強事件也被系統性低估。這說明 full-record HHSA 的平滑結構具有很強的 diagnostic predictability，但不是全部都能在 real time 獲得。

M5 的失敗同樣重要。把更多 branch outputs 交給 fusion network 並不保證改善；在小樣本上，物理重建的結構性限制反而比額外自由度更穩健。

---

## 15. 未來工作

1. 使用 rolling-origin decomposition cache，讓每一個 training sample 都由其當時可見 prefix 建立；
2. 以 median frequency、spectral overlap 和 waveform correlation 進行跨 origin mode matching；
3. 為 phase branch 比較 phasor regression 與 non-negative IF integration；
4. 使用 blocked cross-validation，不只依賴單一 70/15/15 split；
5. 使用多 random seeds 與 calibrated ensembles；
6. 對強 El Niño 使用 weighted loss 或 event-balanced sampling，但不可讓 test events 進入 weighting design；
7. 比較 Ridge、Gaussian process、small FFN 與 residual NN，確定 skill 是否真的來自非線性模型；
8. 加入 spring predictability barrier 的季節分層評估；
9. 分別報告 onset timing、peak timing、peak amplitude 與完整軌跡；
10. 在獨立資料版本與不同 Niño3.4 定義上做 robustness test。

---

## 16. 結論

本研究完成了一套 HHSA Event/Strength 雙分支神經網路。其主要發現為：

1. HHSA state variables 在 offline test 中含有顯著的可預測結構；
2. 顯式分離 phase/timing 與 amplitude/strength，再以 (A\cos\phi) 重建，優於 direct NN 與 single-state HHSA network；
3. individual AM-IMF networks 使不同調制尺度可以使用不同 history length；
4. fusion ResNet 沒有改善，且明顯 overfit；
5. offline 的高 skill 沒有延續到 causal El Niño peak hindcast；
6. 目前最主要障礙是 endpoint effects、mode identity instability 與強事件振幅低估；
7. 因此此架構目前適合作為可解釋的研究與診斷工具，尚不能視為 operational ENSO forecast system。

---

## 參考文獻

[1] Sohail, T., Zika, J. D., & Ehmen, T. *How accurate are salinity measurements around Antarctica? A machine learning based approach*. **Machine Learning: Earth**. DOI: [10.1088/3049-4753/ae7113](https://doi.org/10.1088/3049-4753/ae7113).

[2] Sohail, T. (2024). *Machine learning-based quality assessment of Antarctic margins salinity – code, data and figures* (Version 1.0) [Dataset and software]. Zenodo. DOI: [10.5281/zenodo.14010532](https://doi.org/10.5281/zenodo.14010532).

[3] Huang, N. E., Shen, Z., Long, S. R., et al. (1998). The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis. *Proceedings of the Royal Society A*, 454, 903–995. DOI: [10.1098/rspa.1998.0193](https://doi.org/10.1098/rspa.1998.0193).

[4] Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv: [1412.6980](https://arxiv.org/abs/1412.6980).

### 引用說明

Sohail 等人的引用用來承認 state-vector representation、cyclic sine/cosine encoding、FFN/residual FFN 與其公開程式所提供的實作啟發。HHSA decomposition、Event/Strength 雙分支、individual AM-IMF networks、(A\cos\phi) physical synthesis、ENSO hindcast 和 causal leakage analysis 均為本專案針對 Niño3.4 問題所做的改造，不應歸因於 Sohail 原論文。

## 17. Raw、EMD-only 與 Full HHSA 的 matched ablation

為單獨檢驗 HHSA 的增量價值，本實驗排除第二層 AM-IMFs，並比較 Raw、EMD-only、Full HHSA。三者經 training-only standardization/PCA 統一為 60 維，再進入完全相同的 Sohail Dense residual network：Dense(64) 加三個 256→128→64 residual blocks。Offline 使用五個 seeds；causal 使用三個 seeds，先在每個事件內平均後再以四個獨立事件計算 skill。

### Offline mean R²（五個 seeds）

| Lead | Raw | EMD-only | Full HHSA |
|---:|---:|---:|---:|
| 1 | 0.554 | **0.805** | 0.568 |
| 3 | 0.286 | **0.763** | 0.581 |
| 6 | -0.100 | **0.646** | 0.470 |
| 9 | -0.257 | **0.653** | 0.348 |
| 12 | -0.303 | **0.591** | 0.325 |

### Causal El Niño peak RMSE（四個獨立事件）

| Lead | Raw | EMD-only | Full HHSA |
|---:|---:|---:|---:|
| 3 | **0.947** | 1.462 | 1.388 |
| 6 | **1.359** | 1.766 | 1.654 |
| 9 | 1.710 | 1.790 | **1.579** |

結果表示：EMD 分量在 full-record offline setting 中提供很強的表示優勢；加入 IA、IF 與 phase 並沒有再勝過 EMD-only。因果事件測試中，Raw 在 3、6 月最好，Full HHSA 只在 9 月略好。因此目前沒有證據證明 Full HHSA 能穩定改善即時預報；offline 的 EMD 優勢可能部分來自 full-record decomposition。

輸出：`results/raw_emd_hhsa_offline_summary.csv`、`results/raw_emd_hhsa_causal_summary.csv`、`figures/07_raw_emd_hhsa_offline_ablation.png`、`figures/08_raw_emd_hhsa_causal_ablation.png`。
