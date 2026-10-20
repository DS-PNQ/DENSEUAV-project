# Đánh giá thực nghiệm các cải tiến cho định vị UAV–vệ tinh trên DenseUAV: descriptor tích phân ring×sector, ràng buộc nhất quán không gian mềm và chẩn đoán lỗi

*Báo cáo kỹ thuật nội bộ, tổng hợp từ `baseline06.ipynb` (thí nghiệm A) và `baseline06v3.ipynb` (thí nghiệm B). Kaggle, 2×T4, 1 seed mỗi cấu hình.*

---

## Tóm tắt

Em đánh giá ba hướng cải tiến cho bài toán truy hồi ảnh drone → vệ tinh trên DenseUAV (Dai et al., TIP 2024): (CT1) descriptor tích phân Lebesgue trên lưới ring×sector đặt trong head mạng; (CT2) ràng buộc nhất quán không gian cục bộ dạng mềm (loss thứ tự, nhãn làm mềm theo GPS, làm mượt điểm theo lân cận GPS); (CT3) khung chẩn đoán lỗi theo khoảng cách địa lý. Hai pipeline được chạy với hai baseline khác nhau: LPN-B2 (A) và Global ViT-S (B), tổng cộng 15 run huấn luyện và 9 biến thể test-time. Kết quả chính: (i) head chia phần theo vòng tăng +5.9 R@1 so với Global trong cùng pipeline, nhưng vòng tròn có trọng số tích phân chỉ *ngang* LPN (83.74 so với 83.57). Thêm sector và |DFT| để bất biến theo hướng bay còn *giảm* 2.8 R@1. (ii) Mọi dạng bất biến xoay (|DFT| theo sector, TTA xoay 90°) đều làm giảm hoặc không đổi kết quả. (iii) Các ràng buộc làm mượt không gian đạt đúng mục tiêu hình học (Spearman khoảng cách–tương đồng từ −0.37 xuống −0.57, số cặp gián đoạn từ 44 xuống 2), nhưng giảm R@1 tới 2.8–3.0 điểm vì lỗi dịch sang ô kề. (iv) Gộp đặc trưng gallery theo địa điểm (DBA) là cải thiện lớn nhất, +2.4 R@1 / +2.6 SDM@1, đạt **87.43 R@1 / 90.22 SDM@1**. Với cấu hình test-time này, mọi thay đổi lúc huấn luyện đều nằm trong biên nhiễu seed.

---

## 1. Giới thiệu

DenseUAV lấy mẫu vị trí theo lưới dày (khoảng 20 m) trên 14 campus, ảnh drone chụp ở 3 độ cao (80/90/100 m), mỗi vị trí có 6 ảnh vệ tinh (3 tỉ lệ × 2 năm). Lưới dày làm các vị trí kề nhau gần như trùng ảnh. Đây vừa là thách thức chính của bộ dữ liệu, vừa là động cơ cho ba hướng cải tiến:

- **CT1:** biểu diễn hình học–ngữ nghĩa dạng tích phân trên các ô ring×sector, bất biến với hướng bay chưa biết.
- **CT2:** ràng buộc mềm rằng vị trí gần nhau về địa lý thì gần nhau trong không gian đặc trưng, thay cho ràng buộc Lipschitz cứng.
- **CT3:** phân loại lỗi để biết mô hình nhầm ở đâu: ô kề, trong cùng campus, hay giữa các campus.

Câu hỏi nghiên cứu: *mỗi cải tiến có đóng góp vượt nhiễu so với baseline chạy trong cùng pipeline hay không, và nếu có thì nhờ cơ chế nào?*

## 2. Thiết lập thực nghiệm

**Dữ liệu và metric.** Theo giao thức của repo (commit `b8de187`): 2256 lớp train, 777 lớp query test (2331 ảnh drone), gallery 3033 lớp (18 198 ảnh vệ tinh). Metric: Recall@K, AP, SDM@K (trọng số khoảng cách, $s=5\cdot10^3$), thêm MA@20m (top-1 cách vị trí thật < 20 m). Mọi evaluator viết lại đều được `assert` khớp với `results.txt` và `SDM@K.json` do script gốc sinh ra (sai lệch < 0.1 điểm).

**Bảng 1. Hai pipeline.**

| | Thí nghiệm A (`baseline06`) | Thí nghiệm B (`baseline06v3`) |
|---|---|---|
| Baseline | LPN-B2, CE + WeightedSoftTriplet + KL, lr 0.01 | Global (CLS), CE + HardMiningTriplet, lr 0.003 |
| Backbone | ViT-S/16-224 | ViT-S/16-224 |
| Train | SGD, bs 16, 120 epoch, LR ×0.1 ở epoch 70/110 | như A |
| Cache ảnh | chỉ ảnh drone, cạnh ngắn 512 px | mọi ảnh, cạnh ngắn 384 px |
| Kiểm soát nhiễu | thêm seed 777 cho baseline | không |
| Baseline đạt được | 84.77 R@1 / 87.50 SDM@1 | 77.82 R@1 / 81.85 SDM@1 |
| Run trước, độ phân giải đầy đủ | 86.06 / 88.74 | 79.45 / 83.92 |

Mỗi run chỉ đổi đúng một yếu tố so với baseline của pipeline đó. Các tham số test-time ($\sigma$ của GeoSmooth, cấu hình DBA) được cố định *trước* khi xem kết quả test.

## 3. Phương pháp

**CT1: head tích phân ring×sector (IRP ở A, RS ở B).** Gọi $g$ là trường patch token của ViT (hằng trên từng patch) và $U_{kj}$ là ô vòng $k$ × cung $j$ của đĩa tâm ảnh (tâm ảnh ≈ vị trí UAV, camera nhìn thẳng xuống). Mỗi thành phần của descriptor là
$$w_{kj}=\frac{\int_{U_{kj}} f\,g\,dm}{\int_{U_{kj}} f\,dm},\qquad f(x)=\max\!\left(0,\,1-\tfrac{|x|}{R}\right).$$
Tích phân được tính chính xác bằng một ma trận cố định (tổng Riemann 8×8 hoặc 16×16 điểm con mỗi patch). Khi ảnh xoay, các cung dịch vòng. Vì vậy mỗi vòng chỉ giữ $[\bar w_k,\ |\mathrm{DFT}_{1..K}(w_{k\cdot})|]$, bất biến khi xoay $360°/N_\theta$ và khi lật (đã kiểm chứng bằng self-test). Khi $N_\theta=1$, ta được vòng tròn có trọng số $f$, tức một biến thể của LPN.

**CT2: nhất quán không gian mềm.**
- *LC loss (A):* với mỗi mẫu có láng giềng ≤ 30 m trong batch, $\mathcal L=\mathrm{softplus}\big(\alpha(d_F^{\text{near}}-d_F^{\text{far}>60\,m})\big)$. Loss chỉ ép thứ tự gần < xa, không ép hằng số Lipschitz.
- *GeoLS (B):* nhãn CE làm mềm $T=(1-\varepsilon)I+\varepsilon K$, với $K$ là kernel Gauss ($\sigma=20$ m) trên các lớp láng giềng, $\varepsilon=0.1$.
- *GeoSmooth (B, test-time):* $s'(q,g)=\sum_{c'}K_\sigma(c(g),c')\,\bar s(q,c')$, với $\sigma=10$ m. Khi $\sigma=0$, phép này chỉ gộp điểm của 6 ảnh cùng vị trí (*loc-fuse*).

**Lấy mẫu động DSS (A):** theo CEUSP. Từ epoch 20, mỗi nhóm 4 lớp gồm anchor, 1 láng giềng GPS và 2 lớp mà mô hình đang nhầm nhất. Độ tương đồng được đo lại mỗi 6 epoch.

**Test-time (A):** TTA D4 (4 góc × lật) và DBA, tức thay đặc trưng mỗi ảnh vệ tinh bằng trung bình *đã chuẩn hoá* với ảnh cùng ô ở năm còn lại (`DBA years`) hoặc với cả 6 ảnh của địa điểm (`DBA place`).

**CT3: chẩn đoán.** Top-1 sai của mỗi query được xếp vào một loại:
- *ô kề* (≤ 30 m);
- *nhầm trong campus* (> 30 m, cùng thành phần liên thông của đồ thị GPS);
- *nhầm giữa campus*.

Lỗi xa được tách tiếp thành *cặp mơ hồ*, khi hai prototype gần nhau hơn ngưỡng $\tau$ (trung vị khoảng cách từ query tới vị trí đúng), và *lệch miền* trong trường hợp còn lại. Trên prototype gallery, em đo thêm Spearman($d_{geo}$, cos), số cặp gián đoạn (≤ 30 m nhưng ít tương đồng hơn trung vị các cặp cách 60–100 m) và hằng số Lipschitz thực nghiệm $\lVert P_i-P_j\rVert/d_{ij}$.

## 4. Kết quả

### 4.1 Huấn luyện: thí nghiệm A (baseline LPN-B2)

**Bảng 2.** Giao thức repo (TTA lật), sau đó là cấu hình test-time cố định trước `D4 both + DBA years`.

| Run | Thay đổi | R@1 | ΔR@1 | SDM@1 | R@1 (+DBA) | SDM@1 (+DBA) |
|---|---|---|---|---|---|---|
| `lpn` | baseline | 84.77 | 0 | 87.50 | 87.13 | 90.10 |
| `lpn_seed777` | chỉ đổi seed | 84.77 | 0.00 | 87.40 | 86.96 | 89.96 |
| `lpn_dss` | O4: lấy mẫu động | **85.89** | **+1.12** | **88.17** | 87.22 | 90.17 |
| `lpn_ct2` | CT2: sampler GPS + LC loss | 85.33 | +0.56 | 87.97 | 86.83 | 89.82 |
| `irp` | CT1: head IRP (2 ring × 8 sector) | 84.73 | −0.04 | 87.35 | **87.43** | **90.22** |
| `lpn_dss_lc` | O4 + CT2 | 83.44 | −1.33 | 86.70 | 85.37 | 88.84 |
| `dino_lpn` | backbone DINOv2-S/14 | 4.76 | −80.0 | 7.91 | 3.43 | 5.81 |

Ghi chú:
- Hai run ViT-B bị bỏ qua vì vượt ngân sách 12 h.
- IRP có R@10 cao nhất (98.33), thêm 0.39M tham số, MACs không đổi.

### 4.2 Huấn luyện: thí nghiệm B (baseline Global)

**Bảng 3.** Giao thức repo, kèm R@1 theo độ cao bay.

| Run | R@1 | ΔR@1 | SDM@1 | MA@20m | R@1 H80 | R@1 H90 | R@1 H100 |
|---|---|---|---|---|---|---|---|
| `Global` (đối chứng) | 77.82 | 0 | 81.85 | 82.11 | 72.97 | 78.38 | 82.11 |
| `Global_GeoLS` | 74.99 | −2.83 | 80.68 | 81.30 | 72.84 | 74.13 | 77.99 |
| `RS_B2` (2 ring × 8 sector, \|DFT\|₁..₂) | 80.95 | +3.13 | 84.45 | 85.33 | 77.61 | 81.85 | 83.40 |
| `RS_B2_GeoLS` | 77.91 | +0.09 | 82.47 | 82.67 | 75.42 | 79.15 | 79.15 |
| `RS_B2_GeoLS_KL` | 81.38 | +3.56 | 84.81 | 85.07 | 78.89 | 81.47 | 83.78 |
| `LPN_B2` (vòng vuông) | 83.57 | +5.75 | 86.43 | 87.00 | 82.37 | 83.78 | 84.56 |
| `RS_B2_ringonly` (vòng tròn, f decay, 1 sector) | **83.74** | **+5.92** | **86.84** | **87.47** | **83.27** | 83.66 | 84.30 |
| `DINOv2S_RS_B2_GeoLS` | 20.59 | −57.2 | 27.46 | 27.58 | 19.05 | 21.24 | 21.49 |

### 4.3 Test-time

**Bảng 4.** ΔR@1 / ΔSDM@1 so với TTA lật của chính run đó.

| Biến thể | A: `lpn` | A: khoảng trên 6 run | B: `Global` | B: `LPN_B2` | B: `RS_B2_ringonly` |
|---|---|---|---|---|---|
| TTA xoay, chỉ query | −1.72 / −0.82 | −1.4 → −2.7 | −0.04 / +0.76 | −0.99 / −0.28 | −1.59 / −0.82 |
| D4 cả query lẫn gallery | −0.47 / +0.21 | −0.5 → −1.6 | – | – | – |
| **DBA years** (gộp đặc trưng, chuẩn hoá) | **+2.36 / +2.60** | +1.3 → +2.7 | – | – | – |
| DBA place (6 ảnh) | +2.75 / +2.87 | – | – | – | – |
| loc-fuse (gộp *điểm* 6 ảnh) | – | – | −0.34 / −0.36 | −2.53 / −2.10 | −2.62 / −2.44 |
| GeoSmooth σ = 10 m | – | – | −3.39 / −2.43 | −5.53 / −4.33 | −4.80 / −4.04 |

Sweep σ trên run tốt nhất của B, chỉ để phân tích: R@1 = 81.12 (σ = 0 và 5), 78.94 (σ = 10), 65.21 (15), 49.16 (20), 25.35 (30). Kết quả giảm đơn điệu theo σ.

*Lưu ý đo lường.* Với loc-fuse và GeoSmooth, 6 ảnh của một địa điểm có cùng điểm số. Do cách đếm của repo, R@5 = R@1 và AP bị thổi phồng (66.7 → 82.4). Vì vậy các dòng này chỉ nên đọc R@1 và SDM@1. Thí nghiệm A đã bỏ (NaN) các cột này cho đúng.

### 4.4 Chẩn đoán lỗi (CT3)

**Bảng 5.** Phân bố top-1 (% query). A dùng giao thức repo → DBA; B dùng giao thức repo.

| | Đúng | Ô kề ≤ 30 m | Trong campus | Giữa campus | Trung vị sai số (m) |
|---|---|---|---|---|---|
| A `lpn` | 84.77 → 87.13 | 6.61 → 7.34 | 5.58 → 3.69 | 3.04 → 1.84 | 43.0 → 21.3 |
| B `Global` | 77.82 | 9.70 | 7.46 | 5.02 | – |
| B `Global_GeoLS` | 74.99 | **14.46** | 6.74 | 3.82 | – |
| B `RS_B2_ringonly` | 83.74 | 7.68 | 5.83 | 2.75 | – |

**Bảng 6.** Hình học của embedding trên prototype vị trí (B).

| Run | Cặp mơ hồ (> 100 m, cos ≥ τ) | Cặp gián đoạn (≤ 30 m) | Spearman($d_{geo}$, cos) | Lipschitz p95 (1/m) |
|---|---|---|---|---|
| Global | 75 | 44 | −0.37 | 0.0508 |
| Global_GeoLS | 87 | **2** | **−0.57** | **0.0427** |
| RS_B2 | 24 | 60 | −0.34 | 0.0521 |
| RS_B2_GeoLS | 34 | 2 | −0.55 | 0.0442 |
| LPN_B2 / RS_B2_ringonly | 24 / 24 | 68 / 73 | −0.33 | 0.0528 |

Ở A, chỉ 26–34% lỗi xa thuộc loại *cặp mơ hồ*. Khoảng 2/3 còn lại là *lệch miền*: hai vị trí vốn tách được trên bản đồ vệ tinh, nhưng đặc trưng của query bị lệch.

So sánh từng query ở cấu hình DBA giữa `lpn` và `lpn_seed777` cho thấy 95 query chuyển từ đúng sang sai và 91 query chuyển từ sai sang đúng, tức khoảng 4% query đổi kết quả chỉ vì seed. Giữa `lpn` và `lpn_dss` con số là 82 và 84; giữa `lpn` và `irp` là 86 và 93. Các mức này không khác mức của seed.

## 5. Thảo luận

**(1) Phần lớn lợi ích nằm ở cấu trúc vòng, không ở sector hay trọng số tích phân.** Head chia phần theo vòng tăng khoảng +5.9 R@1 so với Global. Lợi ích tập trung ở độ cao thấp: +10.3 ở H80 so với +2.2 ở H100 (Bảng 3), đồng thời giảm cặp mơ hồ từ 75 xuống 24. Tuy vậy, vòng tròn với $f$ decay (83.74) và vòng vuông LPN (83.57) cho kết quả như nhau trong biên nhiễu. Như vậy, ở dạng tối giản nhất, CT1 *tái hiện* LPN chứ chưa vượt được nó. IRP ở A cũng không khác LPN (−0.04).

**(2) Bất biến xoay gây hại trên DenseUAV.** Ba bằng chứng độc lập cùng chiều:
- thêm sector + |DFT| làm giảm 2.8 R@1 (RS_B2 so với ringonly);
- TTA xoay giảm R@1 ở cả 11/11 run dùng head chia phần (A và B);
- IRP (vốn bất biến xoay) không hơn LPN.

Giả thuyết hợp lý nhất: hướng bay của query trong DenseUAV không phân bố đều mà gần thẳng hàng với ảnh vệ tinh (hướng bắc). Khi đó, ép bất biến xoay làm mất thông tin hướng, vốn có ích cho việc phân biệt. Giả thuyết này cần được kiểm tra trực tiếp trên metadata hoặc bằng ước lượng hướng của ảnh drone, trước khi đưa "rotation invariance" vào bài báo như một ưu điểm.

**(3) Ràng buộc nhất quán không gian mềm đạt mục tiêu hình học nhưng không cải thiện truy hồi.** GeoLS làm embedding mượt hơn rõ rệt (Bảng 6), nhưng chuyển lỗi sang ô kề: tỉ lệ ô kề tăng từ 9.7% lên 14.5%, trong khi lỗi giữa campus giảm từ 5.0% xuống 3.8%. Kết quả là R@1 giảm 2.8. SDM@1 và MA@20m chỉ giảm 1.2 và 0.8 vì các lỗi mới đều nằm gần vị trí đúng. GeoSmooth gây hại đơn điệu theo σ. LC loss ở A có hiệu ứng nhỏ hơn nhiễu (+0.56), và còn làm giảm 1.33 khi ghép với DSS. Kết luận: trên lưới 20 m, R@1 thưởng cho khả năng phân biệt *sắc* giữa các ô kề, nên mọi prior làm mượt cục bộ đều đối nghịch với metric chính. Nếu muốn giữ CT2, nên đánh giá bằng metric theo khoảng cách (SDM, MA@m), hoặc dùng nó để giảm lỗi giữa campus thay vì để tăng R@1.

**(4) Gộp đặc trưng gallery là cải thiện lớn nhất, và cách gộp quyết định kết quả.** DBA (trung bình đặc trưng rồi chuẩn hoá lại) cho +1.3 đến +2.7 R@1 trên mọi run của A. Nó giảm gần nửa lỗi xa (8.6% → 5.5%) và giảm trung vị sai số từ 43 m xuống 21 m. Ngược lại, loc-fuse ở B (trung bình *điểm số*, tức $q\cdot\bar g$ không chuẩn hoá) lại giảm R@1. Khác biệt gần như chắc chắn đến từ chuẩn hoá: $q\cdot\bar g=\lVert\bar g\rVert\cos(q,\bar g)$ phạt những địa điểm có 6 ảnh vệ tinh khác nhau nhiều. DBA chỉ dùng siêu dữ liệu sẵn có của bản đồ (ảnh nào thuộc ô nào), nên hợp lệ khi triển khai. Tuy nhiên, khi so với SOTA công bố, vốn không dùng DBA, cần báo cáo tách riêng.

**(5) Lấy mẫu động là thay đổi huấn luyện hứa hẹn nhất, nhưng chưa đủ bằng chứng.** DSS cho +1.12 R@1 / +0.67 SDM@1 ở giao thức repo, cùng chiều với CEUSP. Với 1 seed, khi đã có DBA, chênh lệch thu hẹp còn +0.09, tức 2 query. Hai seed của baseline trùng nhau về R@1 tổng nhưng vẫn có khoảng 4% query đổi kết quả. Vì vậy chênh lệch dưới khoảng 1 điểm chưa thể kết luận.

**(6) Tích hợp DINOv2 thất bại ở cả hai pipeline** (4.76 R@1 với lr 0.01; 20.59 với lr 0.003). Lỗi chủ yếu là nhầm giữa campus (63% ở A), tức đặc trưng gần như không mang thông tin. Việc lr nhỏ hơn cho kết quả tốt hơn gợi ý nguyên nhân nằm ở tối ưu hoá (SGD với lr của ViT-S). Kết quả này không phản ánh năng lực của DINOv2: tài liệu báo cáo 86.27 R@1 với DINOv2-B.

## 6. Hạn chế

- Mỗi cấu hình chỉ chạy 1 seed; chỉ A có một cặp seed để ước lượng nhiễu.
- Cả hai baseline đều thấp hơn run độ phân giải đầy đủ trước đó: −1.29 R@1 ở A, −1.63 ở B. Chưa tách được phần do cache resample ảnh và phần do nhiễu.
- A và B khác nhau ở cả baseline, loss, cache và định nghĩa vùng (gap 200 m ra 13 vùng; bán kính 100 m ra 14 site). Do đó chỉ nên so sánh trong cùng một thí nghiệm, không so chéo.
- ViT-B (O2) chưa chạy được vì vượt ngân sách 12 h.
- Ở B, R@5/AP của các dòng gộp gallery chưa được sửa theo cách đếm khi có điểm trùng.

## 7. Kết luận và hướng tiếp theo

Trên DenseUAV, kết quả tốt nhất thu được là **87.43 R@1 / 90.22 SDM@1** (IRP + D4 + DBA, ViT-S). Cải thiện này đến chủ yếu từ (a) head chia phần theo vòng và (b) gộp đặc trưng gallery theo địa điểm. Hai thành phần đặc trưng của CT1 và CT2 không cải thiện R@1 trong các thí nghiệm này: bất biến xoay qua sector/|DFT| và làm mượt không gian. CT3 cho thấy lỗi chủ yếu nằm ở ô kề, và phần lớn lỗi xa thuộc loại lệch miền chứ không phải cặp mơ hồ.

Đề xuất theo thứ tự ưu tiên:
1. Chạy 3 seed cho `lpn`, `lpn_dss` và `irp`, báo cáo trung bình ± độ lệch chuẩn.
2. Kiểm tra phân bố hướng bay của query, để xác nhận hoặc bác bỏ giải thích (2).
3. Ở B, sửa loc-fuse thành trung bình đặc trưng có chuẩn hoá, rồi đo lại để xác nhận (4).
4. Định vị lại CT2 như một công cụ giảm lỗi giữa campus, đánh giá bằng SDM và MA@20m.
5. Sửa công thức tối ưu cho DINOv2 (AdamW, lr nhỏ, layer-wise decay) trước khi kết luận về backbone.

## Tài liệu tham khảo

1. Dai M. et al. *Vision-Based UAV Self-Positioning in Low-Altitude Urban Environments* (DenseUAV). IEEE TIP, 2024.
2. Wang T. et al. *Each Part Matters: Local Patterns Facilitate Cross-View Geo-Localization* (LPN). IEEE TCSVT, 2021.
3. CEUSP, arXiv:2502.11408, 2025.
4. Yang et al., IEEE RA-L, 2025, DOI 10.1109/LRA.2025.3527762.
5. G2CL, arXiv:2502.16164, 2025.
6. Arandjelović R., Zisserman A. *Three things everyone should know to improve object retrieval* (DBA). CVPR, 2012.
7. Nhóm tác giả. *Lebesgue Integral-based Shape Vector Descriptors for GPS-Denied UAV Localization* (bản thảo).
