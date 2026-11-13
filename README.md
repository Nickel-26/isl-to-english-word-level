# INCLUDE-50 live ISL → English translator (course project)

Word-level Indian Sign Language recognition. A webcam recognizes **one isolated
sign at a time** and displays the English word. No sentence building, no grammar.

Pipeline: **MediaPipe keypoints → shoulder-normalized 32-frame sequence → BiGRU
→ live webcam demo.** The sign list lives only in `configs/include50.yaml`.

## Dataset facts (verified 2026-11-12)

| Item | Value |
|---|---|
| Official source | [Zenodo record 4010759](https://zenodo.org/records/4010759) (CC-BY-4.0) |
| Archive size | 44 ZIP parts, 56.8 GB compressed |
| INCLUDE-50 clips | **958** = 689 train + 77 val + 192 test |
| Per-class clips | 14–25 |
| Folder layout | `Category/NN. Sign/MVI_xxxx.MOV` |
| Video sample | 1920×1080, 25 fps, ~67 frames (2.7 s) |

Notes:

- There is **no separate INCLUDE-50 archive**. The 50 signs are selected by the
  official split files. They span all 15 categories, but only **35 of the 44
  ZIPs** contain them.
- The split and the ZIPs disagree on extensions for some clips (split says
  `.MOV`, archive holds the same recording as `.MP4`). Files are matched by
  folder + filename stem, not by extension. 15 split paths also sit in an
  `Extra/` subfolder; the sign label always comes from the second path
  component (`Category/Sign/...`).
- The Google Drive split link in the Zenodo record is **404**. The split files
  used here come from
  [AI4Bharat's official inference repo](https://github.com/AI4Bharat/INCLUDE-MS-Teams-Integration/tree/main/src/inference/INCLUDE/train_test_paths)
  and have the same 689/77/192 counts.
- Label rule: keep letters only, lowercase. `54. Cell phone → cellphone`,
  `46. you (plural) → youplural`.

## Layout

```
configs/include50.yaml     sign list + all settings (single source of truth)
src/isl/                   config, splits, HTTP, remote ZIP, archive index,
                           keypoint extraction, dataset/augmentation, models
scripts/                   fetch_splits, fetch_models, index_archives,
                           download_data, verify_dataset, check_env,
                           extract_keypoints, train, evaluate, demo
data/raw/                  Category/NN. Sign/*.MOV (958 files, ~14 GB)
data/splits/               include50_{train,val,test}.txt
data/manifests/            archive_index.json, dataset_report.json, logs
data/keypoints/            one cached .npz per clip (32 × 137 float32)
models/mediapipe/          hand_landmarker.task, pose_landmarker_lite.task
models/checkpoints/        trained classifier + training history
reports/                   evaluation reports and live-demo session logs
```

`data/` and `models/` are gitignored.

## Setup

The repository already has a working `.venv` (Python 3.14, CUDA PyTorch,
MediaPipe 1.2.2 Tasks API). To recreate it, see `requirements.txt`.

```bash
cd /home/pranay/Pranay/Coding/ivp
source .venv/bin/activate

python scripts/check_env.py --smoke-test   # versions, CUDA, MediaPipe models
python scripts/fetch_models.py             # ~13 MB
python scripts/fetch_splits.py            # 689/77/192 split files
python scripts/index_archives.py          # reads ZIP tables of contents only
```

## Stage 1: download and verify the videos

```bash
# See the exact download plan without downloading anything:
python scripts/download_data.py --dry-run

# Default: fetch only the needed ZIP members over HTTP ranges.
# ~940 clips / ~12 GB compressed, resumable, CRC-checked per file.
# In another terminal, watch live progress:
#   python scripts/watch_download.py        # updates download_progress.txt
#   watch -n 2 cat download_progress.txt
python scripts/download_data.py

# Fallback: download whole ZIPs (~42 GB), extract only the listed clips,
# then delete each ZIP.
python scripts/download_data.py --mode archive

# Check counts, labels, overlaps, and decode a few videos:
python scripts/verify_dataset.py --probe 10
```

Expected result: **958 / 958 clips present**, 50 signs, no missing files, no
overlaps. `download_data.py` can be re-run safely: it skips clips already on
disk and retries only what is missing. Use `--connections N` to tune the number
of parallel HTTP range requests (default 16).

Disk use: the extracted videos are ~14 GB. Remote mode never stores a full ZIP;
archive mode keeps at most one ZIP (~1.8 GB) at a time and checks free space
before each one.

## Stage 2: extract keypoints

MediaPipe pose (25 upper-body points, no face) + both hands (21 points each),
x/y only, shoulder-relative normalization, resampled to 32 frames. Each frame
is 137 numbers: 25×2 pose + 21×2 left hand + 21×2 right hand + 3 presence flags
(pose, left hand, right hand). Hands are assigned to left/right by proximity to
the pose wrists, not by MediaPipe's handedness label.

```bash
# Extract everything (resumable; valid cached clips are skipped).
# In another terminal, watch live progress:
#   watch -n 2 cat keypoints_progress.txt
python scripts/extract_keypoints.py --workers 4

# Small trial run first if you prefer:
python scripts/extract_keypoints.py --limit 10

# Check the cache without touching MediaPipe:
python scripts/extract_keypoints.py --verify-only

# Re-extract a single sign, or force everything:
python scripts/extract_keypoints.py --only hello --overwrite
```

Expected result: **958 / 958** `.npz` files, each `(32, 137)`, no NaN/inf.
Extraction is CPU-bound (~2 s/clip per worker); the run is safe to interrupt and
restart. Cache files record `sign_hash` and `cache_version`, so changing the sign
list or the feature layout invalidates the stale entries automatically.

## Stage 3: train the classifier

```bash
# Full run. Writes training_progress.txt, the checkpoint and a history JSON.
python scripts/train.py

# In another terminal:
watch -n 2 cat training_progress.txt

# Fast sanity check:
python scripts/train.py --epochs 2

# Switch the architecture in configs/include50.yaml:
#   model.type: bigru        (default)
#   model.type: transformer  (optional)
```

What it does:

- Loads the 689 train and 77 val clips from the cache. The official 192 test
  clips are untouched until Stage 4.
- Adds **600 synthetic rest sequences** to train and **120** to val: the median
  pose of a real clip, hands placed at the hips, a slow sway, and random
  hand-detection dropouts. INCLUDE has no real "no sign" clips.
- Applies random **time shift** (±2 frames), **scale** (0.9–1.15) and **speed**
  (0.8–1.25) augmentation with probability `augment.p`.
- Uses class-balanced cross-entropy (so 600 rest rows do not swamp 689 sign
  rows), label smoothing, AdamW, gradient clipping, a learning-rate scheduler
  and early stopping.
- Picks the checkpoint by **balanced validation accuracy** = mean(sign-only
  accuracy, rest accuracy), so a model that always says "No sign" cannot win.

Outputs: `models/checkpoints/include50_best.pt`, `include50_best_last.pt` and
`include50_best_history.json`.

Device note: Stage 2 runs on the **CPU** (MediaPipe/TFLite); Stage 3 trains on
the **RTX 3050 via CUDA** when available.

Training result (2026-11-12): 60 epochs in 20 s (early stopping), best epoch 48.
Validation balanced score **0.9828**: sign accuracy **0.974** (75/77) and
synthetic-rest accuracy **0.992**. The model has 515k parameters.

## Stage 4: evaluate on the official test split

```bash
python scripts/evaluate.py

# Skip the synthetic rest sanity check:
python scripts/evaluate.py --rest-check 0
```

Outputs in `reports/`:

| File | Contents |
|---|---|
| `test_metrics.json` | accuracy, top-3, macro F1, published comparison, rest check |
| `confusion_matrix.png` / `.csv` | 51-class confusion matrix |
| `per_class_accuracy.csv` | recall + support per sign, worst first |
| `most_confused.csv` | top off-diagonal confusions |
| `classification_report.txt` | sklearn precision/recall/F1 table |

Measured result (2026-11-12):

| Metric | Value |
|---|---|
| Top-1 accuracy (192 test clips) | **96.88%** (186/192) |
| Top-3 accuracy | 98.44% |
| Macro F1 | 0.9678 |
| Published INCLUDE-50 | 94.5% |
| Synthetic rest check | 98.3% of 300 predicted as rest |

The published **94.5%** uses a pretrained CNN feature extractor plus an encoder,
not MediaPipe keypoints, so the comparison is **not like-for-like**. The split is
also random videos rather than signer-independent, which likely makes both
numbers optimistic. Do not claim a state-of-the-art result.

Worst signs on the test split: *Boy* 2/4, *Big / Large* 3/4, *Cell phone* 3/4.
Most confusions: *Boy → Girl* (2), then one-off pairs such as
*Big / Large → Loud*, *Brother → Quiet*, *Cell phone → Black*.

## Stage 5: live webcam demo

```bash
# Default camera:
python scripts/demo.py

# Other camera or a recorded clip:
python scripts/demo.py --source 1
python scripts/demo.py --source path/to/clip.MOV --headless

# Tune the gate without editing the config:
python scripts/demo.py --threshold 0.7 --min-streak 3 --alpha 0.4
```

How it works:

- Every camera frame goes through the **same** MediaPipe pose + hands pipeline
  and shoulder normalization used for extraction.
- Frames are kept in a **2.5 s timed buffer**. The first prediction waits until
  the buffer has ~2 s of video (early partial windows are not representative).
- Every 0.33 s the buffer is resampled to 32 frames and classified.
- Class probabilities are smoothed with an **EMA** (`alpha`), and a word is
  only shown after **3 consecutive** smoothed predictions above the **0.6**
  confidence threshold. Otherwise the display stays at `...`.
- The `rest` class displays **No sign**. Press `q` or `Esc` to quit.

Verified on dataset clips: *Court* 0.93 and *Hello* 0.86 (using `--min-streak 2`
because a 2.4 s clip only yields two full-window predictions). A 6 s real-camera
headless check ran at **15 FPS**; with no hands in view it produced zero
committed labels, which is the intended behaviour of the gate.

Every session writes `reports/demo_session_<timestamp>.json` with FPS, frames
without hands/pose, low-confidence counts, label switches and a per-inference
timeline. That log is the input for the **separate live-demo failure analysis**.

### Live-demo failure analysis (separate from test accuracy)

These are demo-path observations, not test-set metrics:

| Scenario | Observed | Interpretation |
|---|---|---|
| Dataset clip *Court* | 0.91–0.93 confidence, correct | Windowed preprocessing matches training |
| Dataset clip *Hello* | 0.86 confidence, correct | Works with a different sign/category |
| 2.4 s clip, default `min-streak 3` | Only 2 full-window inferences, display stays `...` | Expected: the streak needs ~1 s of held pose after the buffer fills |
| Real camera, no hands in view (6 s) | 79/79 frames without hands, 10/10 inferences below threshold, 0 committed labels | Confidence gate correctly rejects unknown input |
| Live loop FPS | ~15 FPS at 640×480 on the laptop CPU | Meets the 15+ FPS target; MediaPipe runs on CPU |

Failure modes to quantify in a real signing session using the JSON log:

- **No/occluded hands:** high `frames_without_hands`; expect low confidence.
- **Distance/lighting:** low confidence but plausible raw labels.
- **Flicker:** high `label_switches` or short `streak` values.
- **Rest false positives:** `rest_predictions` is low when nothing is signed,
  but the synthetic rest class may not cover leaning, walking or one hand only.
- **Signer mismatch:** INCLUDE has few signers; a new person will score below
  the 96.9% test number. This is the main expected live failure.

## Design choices (plain language)

- **Fetch only what we need:** each video is a separately compressed member of
  a remote ZIP, so we download just those members with HTTP ranges instead of
  42 GB of archives. CRC-32 + size checks catch any corruption.
- **Keypoints, not pixels:** tiny input, fast, privacy-friendly, works on CPU.
- **Shoulder normalization:** subtract the shoulder midpoint and divide by
  shoulder width, so camera distance/position does not matter.
- **Hand identity** comes from proximity to the pose wrists, not MediaPipe's
  handedness label (which is ambiguous with mirrored webcams).
- **32 frames per clip:** enough to see the whole sign, small model.
- **Time-based live window (2.5 s):** matches the ~2.7 s dataset clips, so
  training and demo see the same temporal scale.
- **BiGRU first:** sequence order matters; it is small and fast. A small
  Transformer is config-selectable later.
- **Rest class + confidence gate + smoothing** prevent flicker and false words.
- **No hard-coded 50:** class count, label order and display names are derived
  from the config. Switching to 263 signs is a config/split change only.

## Status

- [x] Stage 1: skeleton, split fetcher, archive indexer, downloader, verifier
  - Verified 2026-11-12: 958/958 clips, 13.0 GB, 689/77/192, no missing or
    extra files, 10/10 OpenCV probes OK.
- [x] Stage 2: keypoint extraction (cached `.npz`)
  - Verified 2026-11-12: 958/958 clips, `(32, 137)`, 689/77/192,
    16 MB cache, no NaN/inf, presence flags valid.
- [x] Stage 3: training (BiGRU, augmentation, rest class)
  - Verified 2026-11-12: 60 epochs (early stop, 20 s), best epoch 48,
    val balanced score 0.9828 (sign 0.974, rest 0.992), 515k parameters.
- [x] Stage 4: evaluation (confusion matrix, published comparison)
  - Verified 2026-11-12: test top-1 0.9688, top-3 0.9844, macro F1 0.9678;
    synthetic rest 98.3%; reports written to `reports/`.
- [x] Stage 5: live webcam demo
  - Implemented and verified on dataset clips and a 6 s real-camera headless
    run (~15 FPS). A meaningful webcam accuracy number still needs a person
    signing in front of the camera; session logs support the failure analysis.

## Known caveats

- The published **94.5%** INCLUDE-50 result uses a *pretrained CNN feature
  extractor + encoder*, not keypoints alone. It is not directly comparable.
- The split is random videos, not signer-independent, so live accuracy will be
  lower than test accuracy.
- INCLUDE has no idle/rest videos. The rest class is synthesised from real poses
  (median pose, hands at the hips) and therefore does not cover every real
  no-sign situation. The confidence gate and streak requirement compensate.
- A short recorded clip may end before three full-window predictions arrive, so
  the display can stay at `...`. This is expected; a live signer holds the pose
  long enough. Use `--min-streak 2` when testing on short clips.
- Tests: `python -m unittest discover -s tests` (30 tests, no camera needed).
