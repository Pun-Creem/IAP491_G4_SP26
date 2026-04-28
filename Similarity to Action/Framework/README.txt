Phân tích mẫu malware từ CAPE Sandbox report và đề xuất các **D3FEND defensive actions** phù hợp.

---

## Tổng quan

Nhận một file JSON report từ CAPE Sandbox, trích xuất các đặc trưng hành vi (Signatures, TTPs, MBCs), so sánh với dataset gồm 952 mẫu đã biết, rồi đưa ra bảng điểm xếp hạng các hành động phòng thủ theo chuẩn MITRE D3FEND.

```
sample.json  →  Pattern Extraction  →  Similarity  →  Top-K  →  Action Score  →  Ranked Recommendations
```

---

## Cấu trúc thư mục

```
Framework v3/
│
├── framework.py              # Pipeline chính (entry point)
├── config.json               # Default arguments
│
├── 1_Reports/                # 952 CAPE Sandbox JSON reports (dataset)
│
├── 2_Action dataset/         # Dữ liệu action thô
│   ├── actionPerReport.xlsx      # Hash → danh sách actions thực tế
│   ├── unique_actions_dataset.xlsx
│   └── Filter/
│       ├── action-space.xlsx
│       ├── d3fend.csv
│       ├── check_actions.py
│       └── extract_unique_actions.py
│
├── 3_Patterns/               # Pattern space của dataset
│   ├── unique/
│   │   ├── signature_unique.csv  # Danh sách unique signatures
│   │   ├── ttp_unique.csv        # Danh sách unique TTPs
│   │   └── mbc_unique.csv        # Danh sách unique MBCs
│   ├── pattern_signature.csv     # One-hot matrix signatures (952 samples)
│   ├── pattern_ttp_update.csv    # One-hot matrix TTPs (đã cập nhật VT)
│   ├── pattern_mbc_update.csv    # One-hot matrix MBCs (đã cập nhật VT)
│   ├── Unique_pattern.py         # Trích unique features từ pattern CSV
│   └── Pattern extract/
│       ├── PatternExtract.py         # Scan JSON reports → one-hot CSV
│       └── Pattern update VirusTotal/
│           ├── VirusTotalUpdateAPI.py    # Fetch TTP/MBC từ VT API
│           └── CascadeUpdate.py         # Merge VT data vào pattern CSV
│
├── 4_Samples/                # Scripts xử lý sample mới
│   ├── Pattern_Sample.py         # Trích pattern từ sample JSON (dùng pattern space cố định)
│   ├── VirusTotalUpdateAPI.py
│   ├── CascadeUpdate.py
│   └── unique Pattern space/     # Copy pattern space dùng cho sample
│
├── 5_Similarity/             # Tính similarity
│   ├── sim.py                    # Jaccard / Cosine similarity
│   ├── sum_similarity.py         # Merge 3 similarity CSV
│   ├── run_all_similarity_and_merge.py  # Chạy cả 3 + merge (GUI)
│   └── dataset pattern/          # Pattern CSVs dùng làm dataset chuẩn
│
├── 6_Action space/           # Mapping TTP → D3FEND actions
│   ├── generate_action_per_ttps.py   # Query D3FEND API tạo mapping matrix
│   ├── build_action_per_ttps.py      # Cập nhật mapping khi dataset mở rộng
│   ├── generate_unique_actions.py    # Trích unique actions + metadata
│   ├── action_per_ttps.csv           # Ma trận TTP × Action (0/1)
│   ├── unique_actions.csv
│   └── d3fend.csv
│
├── 7_Top K/                  # Chọn top-K neighbors
│   ├── top_k_selector.py
│   ├── actionPerReport.xlsx
│   └── unique_actions.csv
│
└── 8_Action Score/           # Tính action score cuối
    ├── map_actions.py            # Map TTP → actions cho 1 file
    ├── map_actions_multi.py      # Map cho nhiều file
    ├── compute_action_score.py   # Tính score cho 1 file
    ├── compute_action_score_multi.py
    └── action_per_ttps.csv
```

---

## Yêu Cầu

```bash
pip install pandas openpyxl requests
```

Python 3.10+ được khuyến nghị.

---

## Sử dụng nhanh

```bash
python framework.py --sample path/to/sample.json
```

Kết quả được lưu tại `output/<sample_name>/`.

### Ví dụ đầy đủ

```bash
python framework.py \
  --sample "4_Samples/981/981_report.json" \
  --metric jaccard \
  --w-sig 0.33 --w-mbc 0.33 --w-ttp 0.34 \
  --top-k 5 \
  --W 3.0 --beta 7.0 \
  --vt-key "your_api_key_here"
```

---

## Cấu hình (`config.json`)

Thay vì truyền arguments mỗi lần, chỉnh `config.json` để lưu default:

```json
{
  "metric": "jaccard",
  "w_sig": 0.33,
  "w_mbc": 0.33,
  "w_ttp": 0.34,
  "top_k": 5,
  "W": 3.0,
  "beta": 7.0,
  "vt_key": "",
  "no_vt": false,
  "out_dir": "",
  "paths": {
    "pattern_sig":    "5_Similarity/dataset pattern/pattern_signature.csv",
    "pattern_ttp":    "5_Similarity/dataset pattern/pattern_ttp_update.csv",
    "pattern_mbc":    "5_Similarity/dataset pattern/pattern_mbc_update.csv",
    "sig_unique":     "3_Patterns/unique/signature_unique.csv",
    "ttp_unique":     "3_Patterns/unique/ttp_unique.csv",
    "mbc_unique":     "3_Patterns/unique/mbc_unique.csv",
    "action_ttps":    "6_Action space/action_per_ttps.csv",
    "action_report":  "7_Top K/actionPerReport.xlsx",
    "unique_actions": "7_Top K/unique_actions.csv"
  }
}
```

**Priority:** `CLI args > config.json > built-in defaults`

---

## Pipeline chi tiết

### Stage 1 — Pattern Extraction

Parse JSON report, trích xuất 3 loại đặc trưng hành vi và encode thành one-hot vector theo pattern space cố định của dataset.

| Input | Output |
|-------|--------|
| `sample.json` | `{stem}_signatture.csv` |
| `signature_unique.csv` | `{stem}_ttp_update.csv` |
| `ttp_unique.csv` | `{stem}_mbc_update.csv` |
| `mbc_unique.csv` | |

### Stage 2 — VirusTotal Enrichment *(tuỳ chọn)*

Query VT Behaviours API để bổ sung thêm TTPs và MBCs mà CAPE có thể bỏ sót. Bỏ qua nếu không có `--vt-key`.

| Input | Output |
|-------|--------|
| SHA256, VT API key | `{stem}_ttp_update_vt.csv` |
| `{stem}_ttp_update.csv` | `{stem}_mbc_update_vt.csv` |

### Stage 3 — Similarity Computation

So sánh vector one-hot của sample với toàn bộ 952 mẫu trong dataset, tính riêng cho từng loại pattern.

| Metric | Mô tả |
|--------|-------|
| **Jaccard** *(default)* | Phù hợp với binary features — đo overlap tập hợp |
| **Cosine** | Phù hợp khi magnitude quan trọng |

| Input | Output |
|-------|--------|
| `{stem}_*.csv` (3 files) | `{stem}_signature_similarity.csv` |
| `pattern_signature.csv` | `{stem}_ttp_similarity.csv` |
| `pattern_ttp_update.csv` | `{stem}_mbc_similarity.csv` |
| `pattern_mbc_update.csv` | |

### Stage 4 — Merge Similarity

Outer join 3 file similarity thành 1 bảng tổng hợp.

```
target_id | dataset_id | similarity_type | signatures_sim | mbcs_sim | ttps_sim
```

### Stage 5 — Top-K Selection

Tính weighted similarity rồi chọn K mẫu tương đồng nhất:

```
weighted_sim = w_sig × sig_sim + w_mbc × mbc_sim + w_ttp × ttp_sim
```

Tra cứu actions thực tế của K mẫu đó từ `actionPerReport.xlsx`.

### Stage 6 — Map TTP → D3FEND Actions

Với mỗi TTP active của sample, tra bảng `action_per_ttps.csv` để lấy D3FEND actions tương ứng. Tạo 2 dạng output:
- **Binary** (`_mapped_action.csv`): action = 0 hoặc 1
- **Duplicate** (`_mapped_action_dupe.csv`): cộng dồn count nếu action xuất hiện ở nhiều TTP

### Stage 7 — Action Score

Tính điểm cho từng action kết hợp thông tin của sample và K neighbors:

```
score(a) = W × p_x(a) + β × Σ(j=1..K) [ sim_j × y_j(a) ]
```

| Ký hiệu | Ý nghĩa |
|---------|---------|
| `p_x(a)` | Action `a` có trong TTP mapping của sample (0/1) |
| `sim_j` | Weighted similarity với neighbor j |
| `y_j(a)` | Action `a` có trong neighbor j (0/1) |
| `W` | Trọng số tự thân (default: 3.0) |
| `β` | Trọng số neighbor (default: 7.0) |

---

## Output files

Tất cả output lưu tại `output/<sample_name>/`:

| File | Nội dung |
|------|---------|
| `{stem}_signatture.csv` | One-hot signatures của sample |
| `{stem}_ttp_update.csv` | One-hot TTPs của sample |
| `{stem}_mbc_update.csv` | One-hot MBCs của sample |
| `{stem}_*_similarity.csv` | Similarity scores (signature / ttp / mbc) |
| `{stem}_merged_similarity.csv` | Merged similarity (3 scores trên 1 dòng) |
| `{stem}_top{K}.csv` | Top-K neighbors + action vectors |
| `{stem}_mapped_action.csv` | D3FEND actions của sample (binary) |
| `{stem}_mapped_action_dupe.csv` | D3FEND actions của sample (count) |
| `{stem}_action_scores.csv` | **Bảng kết quả cuối — ranked actions** |

### Định dạng `_action_scores.csv`

```
rank,action,score
1,D3-HBPI - Homoglyph Based Phishing Identification,12.450000
2,D3-DA - DNS Allowlisting,10.320000
...
```

---

## Cập nhật dataset

### Thêm mẫu mới vào dataset

1. Đặt JSON reports mới vào `1_Reports/`
2. Chạy lại `3_Patterns/Pattern extract/PatternExtract.py` để tái tạo pattern matrix
3. Chạy `3_Patterns/Pattern extract/Pattern update VirusTotal/VirusTotalUpdateAPI.py` để cập nhật VT
4. Chạy `3_Patterns/Unique_pattern.py` để cập nhật unique feature lists
5. Copy 3 file CSV kết quả vào `5_Similarity/dataset pattern/`

### Thêm technique D3FEND mới

```bash
# Cập nhật mapping TTP → actions từ D3FEND API
cd "6_Action space"
python generate_action_per_ttps.py

# Nếu dataset actions mở rộng
python build_action_per_ttps.py
python generate_unique_actions.py
```

---

## Tham số

| Tham số | Type | Default | Mô tả |
|---------|------|---------|-------|
| `--sample` | str | *(bắt buộc)* | Path đến file JSON report |
| `--metric` | str | `jaccard` | Loại similarity: `jaccard` hoặc `cosine` |
| `--w-sig` | float | `0.33` | Trọng số Signature similarity |
| `--w-mbc` | float | `0.33` | Trọng số MBC similarity |
| `--w-ttp` | float | `0.34` | Trọng số TTP similarity |
| `--top-k` | int | `5` | Số neighbors lấy vào Top-K |
| `--W` | float | `3.0` | Hệ số tự thân trong Action Score |
| `--beta` | float | `7.0` | Hệ số neighbor trong Action Score |
| `--vt-key` | str | `""` | VirusTotal API key (bỏ qua Stage 2 nếu trống) |
| `--no-vt` | flag | `false` | Bỏ qua Stage 2 dù có `--vt-key` |
| `--out-dir` | str | `output/<stem>` | Thư mục lưu output |

> `w_sig + w_mbc + w_ttp` phải bằng `1.0`.

---

## Nguồn dữ liệu

| Nguồn | Vai trò |
|-------|---------|
| [CAPE Sandbox](https://capev2.readthedocs.io/) | JSON reports chứa Signatures, TTPs, MBCs |
| [MITRE ATT&CK](https://attack.mitre.org/) | Chuẩn định danh TTPs (T1055, T1059, ...) |
| [MITRE MBC](https://github.com/MBCProject/mbc-markdown) | Malware Behavior Catalog (B0002, C0007, ...) |
| [MITRE D3FEND](https://d3fend.mitre.org/) | Defensive techniques mapping (D3-XXX) |
| [VirusTotal](https://www.virustotal.com/) | Bổ sung TTP/MBC qua Behaviours API |
