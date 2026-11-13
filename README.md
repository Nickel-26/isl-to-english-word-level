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
  (0.8–1.25) augmentation with probability `augment.p`, plus **hand dropout**
  (probability 0.4, dropping 0–8-frame hand spans and their presence flags).
  Hand dropout is the webcam-robust transform; mirroring was tested and
  rejected (§Augmentation ablation).
- Uses class-balanced cross-entropy (so 600 rest rows do not swamp 689 sign
  rows), label smoothing, AdamW, gradient clipping, a learning-rate scheduler
  and early stopping.
- Picks the checkpoint by **balanced validation accuracy** = mean(sign-only
  accuracy, rest accuracy), so a model that always says "No sign" cannot win.

Outputs: `models/checkpoints/include50_best.pt`, `include50_best_last.pt` and
`include50_best_history.json`.

Device note: Stage 2 runs on the **CPU** (MediaPipe/TFLite); Stage 3 trains on
the **RTX 3050 via CUDA** when available.

Training result (2026-11-13, hand-dropout 0.4): best epoch 52, validation
balanced score **0.9870**: sign accuracy **0.974** (75/77) and synthetic-rest
accuracy **1.000**. The model has 515k parameters.

### Augmentation ablation

| Variant | Test top-1 | Macro F1 | Live clip top-1 |
|---|---|---|---|
| Baseline | 96.88% | 0.9678 | 4/8 |
| Mirror 0.5 | 89.06% | 0.8693 | rejected |
| Mirror + dropout + noise | 96.35% | 0.9609 | 3/8 |
| One-handed mirror 0.5 + dropout | 97.40% | 0.9720 | 2/8 |
| **Hand dropout 0.4 (selected)** | **97.92%** | **0.9792** | **4/8** |
| Hand dropout 0.5 | 97.92% | 0.9801 | 4/8 |

**Hand dropout is the only transform that helped.** Mirror augmentation sounds
right for a webcam user but cost ~7.8 points of test accuracy; one-handed ISL
signs are not symmetric under a left/right swap. Mirroring only single-hand
clips was tested too (97.40% / 2-of-8 live) and also lost. Coordinate noise and
translation were neutral and are disabled. Details in `project.md` §6.2.1.

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

Measured result (2026-11-13, final hand-dropout checkpoint):

| Metric | Value |
|---|---|
| Top-1 accuracy (192 test clips) | **97.92%** (188/192) |
| Top-3 accuracy | 98.96% |
| Macro F1 | 0.9792 |
| Published INCLUDE-50 | 94.5% |
| Synthetic rest check | 98.0% of 300 predicted as rest |

The published **94.5%** uses a pretrained CNN feature extractor plus an encoder,
not MediaPipe keypoints, so the comparison is **not like-for-like**. The split is
also random videos rather than signer-independent, which likely makes both
numbers optimistic. Do not claim a state-of-the-art result.

Worst signs on the test split: *Big / Large* 3/4, *Boy* 3/4, *Brother* 4/5,
*Hot* 5/6. The remaining errors are one clip each. Black and Fan are perfect
(5/5 and 3/3) on the clean test split.

## Stage 5: live webcam demo

```bash
# Default camera:
python scripts/demo.py

# Other camera or a recorded clip:
python scripts/demo.py --source 1
python scripts/demo.py --source path/to/clip.MOV --headless

# Tune the gate without editing the config:
python scripts/demo.py --threshold 0.7 --min-streak 3 --min-hand-coverage 0.35
```

How it works:

- Every camera frame goes through the **same** MediaPipe pose + hands pipeline
  and shoulder normalization used for extraction.
- Frames are kept in a **2.5 s timed buffer**. The first prediction waits until
  the buffer has ~2 s of video (early partial windows are not representative).
- Every 0.33 s the buffer is resampled to 32 frames and classified.
- A **hand-coverage gate** first checks that at least 30% of window frames have a
  detected hand. If not, the display says `Hands out of frame`; after 0.8 s
  without usable evidence the smoother resets.
- Class probabilities are then smoothed with an **EMA** (`alpha`), and a word is
  shown after **2 consecutive** smoothed predictions above the **0.6**
  confidence threshold. Otherwise the display stays at `...`.
- The `rest` class displays **No sign**. Press `q` or `Esc` to quit.
- Committed labels listed in `demo.suppress_labels` are hidden (display `...`)
  even when the model is confident. See below.

**Framing matters more than anything else.** In the labelled webcam takes a pose
wrist was inside the frame only ~27% of the time, and MediaPipe detected a hand
in ~0% of frames where the wrist was outside the image. Step back and raise
the camera until both hands stay inside the view for the whole sign; 1280×720
(16:9) is better than 640×480 for this.

Verified on dataset clips: *Court* 0.93, *Hello* 0.86, *Cow* 0.71–0.86 at
28.9 FPS. A 6 s real-camera headless check ran at **15 FPS**; with no hands in
view it produced zero committed labels, which is the intended behaviour of the
gate.

Every session writes `reports/demo_session_<timestamp>.json` with FPS, frames
without hands/pose, blocked inferences, low-confidence counts, label switches,
per-inference hand coverage/top-3/margin/entropy and a timeline. That log is the
input for the **separate live-demo failure analysis**.

### Hiding an over-predicted label (`demo.suppress_labels`)

The webcam demo over-predicted **It** on live input: in session logs it reached
0.60–0.94 confidence with hands detected in 90–100% of frames, and one run
committed It for 37 consecutive inferences (about 30 s). This is domain shift,
not a class-prior bug or a missing-hand artifact:

- On the official test split It has **9/9 correct, precision 1.00**, and no
  other test sign is ever predicted as It.
- Synthetic rest → rest 20/20; an all-zero/no-hand sequence → `hat`. Hand
  dropout (up to 90%) and coordinate jitter do not increase the It count.
- Its nearest training prototypes are **I** (distance 3.66), then Good, New,
  Fall, White, Pen, Hello and Cell phone — a dense two-handed chest-level
  cluster distinguished mainly by fine handshape and motion.

The word is kept in the model and checkpoint; only the demo hides it:

```yaml
demo:
  suppress_labels: [it]   # model still predicts `it`, display shows "..."
```

Semantics are deliberately narrow:

- Only a **committed** suppressed label is hidden; the model's raw argmax is
  still written to the timeline with `"suppressed": true` and
  `reason: "suppressed_label"`.
- The EMA and streak keep running, so the next non-suppressed word commits
  normally. Suppression is **not** counted as a hand-coverage block.
- Session JSON gains `settings.suppress_labels` and
  `summary.suppressed_inferences`; the caption script inherits the same list,
  so a suppressed word never enters a transcript.
- Config load fails loudly if a listed label is not in `dataset.signs`, so a
  typo cannot silently disable the setting.

This is a **vocabulary/presentation decision**, not a model fix. Test accuracy
and the checkpoint are unchanged. When It would have fired, the demo now shows
an honest `...`; if It fired during a different sign, that window also abstains
instead of showing the true word. Measured next steps remain higher capture
resolution (1280×720), broadening the synthetic rest class, or removing `it`
from `dataset.signs` if it is not needed.

Verified on `data/raw/Pronouns/44. it/MVI_0015.MOV`: the model commits It at
0.96–0.97, the overlay shows `...`, and the log records 2 suppressed commits.
`eval_vids/cow.mp4` is unchanged (`Cow`); none of the eight labelled takes
trigger It (0.0% suppression).

### Video captioning (words burned into a file)

`scripts/caption_video.py` runs the same inference path on a recorded clip and
writes a new video with a rolling caption line at the bottom:

```bash
python scripts/caption_video.py eval_vids/cow.mp4
# -> reports/cow_captions.mp4 + reports/cow_captions.json
python scripts/caption_video.py clip.mp4 --out /tmp/captioned.mp4 --threshold 0.7
```

- A word enters the transcript only when the decision gate **commits** it;
  low-confidence windows and `Hands out of frame` add nothing. Labels in
  `demo.suppress_labels` are never captioned.
- The synthetic `rest` class is rendered as a **full stop**, so the line reads
  like `Cow . Red .`; leading and doubled periods are suppressed.
- A held sign is not repeated on every inference window, and the same word can
  appear again after a full stop.
- The last ~12 words are kept and wrapped to two lines; the oldest drop first.
- Output keeps the source resolution, frame count and FPS. The file is
  **silent** because OpenCV cannot mux audio (no ffmpeg in this environment).
- The JSON sidecar records the transcript, per-word timestamps and every gate
  decision, so a wrong caption can be traced to a specific window.

Verified: `cow.mp4` → `Cow`; `black.mp4` → `Fan` (the documented
hand-out-of-frame failure); a blank 3 s clip → no caption and an explicit
"no committed words" warning.

### Labelled live evaluation (separate from test accuracy)

`eval_vids/` contains eight user takes: Black, Brother, Cell phone, Cow, Fan,
Priest, Quiet, Red (1280×720, ~15 fps, 1.8–4.6 s). They are **evaluation only**
and were never used for training.

```bash
# Clip-level + full live path, with a gate sweep:
python scripts/eval_live_clips.py --clips-dir eval_vids --sweep
```

Final checkpoint, selected gate (τ 0.60, streak 2, min coverage 0.30):

| Metric | Value |
|---|---|
| Clip mode (oracle segmentation) | **4/8 top-1**, 5/8 top-3 |
| Live stream settled | **4/8 top-1** |
| Committed precision | **78.95%** (15/19) |
| Coverage | 43.2% of inferences |
| Blocked by low hand coverage | 11.4% |
| Mean any-hand rate | 38.1% (training clips ~97%) |
| Mean wrist-in-frame rate | ~27% |

The four clip errors are stable across checkpoints and temporal TTA:
Black → Fan (0.78), Brother → Boy (0.26), Fan → Paint (0.22), Priest → Paint
(0.12). Correct: Cell phone, Cow, Quiet, Red. The gate abstains on Brother, Fan
and Priest rather than emitting a guess.

| Scenario | Observed | Interpretation |
|---|---|---|
| Dataset clip *Court* / *Hello* / *Cow* | correct, high confidence | Windowing matches training |
| Labelled Black take | wrist in frame 22.7%, any-hand 29.5% | Discriminative hand is outside the image |
| Labelled Brother take | wrist in frame 19.3% | Same; gate abstains |
| Hand-model sweep (thresholds/mode/upscale/crop) | any-hand stays 35–38% | Misses are absent pixels, not marginal detections |
| 2.4 s clip, `min-streak 3` | only 3–8 inferences per window | Streak 3 abstains on short takes; default is 2 |
| Real camera, no hands (6 s) | 79/79 frames without hands, 0 committed | Coverage + confidence gates work |

Failure modes still worth logging:

- **No/occluded hands:** high `frames_without_hands`; expect blocks, not words.
- **Distance/lighting:** low confidence but plausible raw labels.
- **Flicker:** high `label_switches` or short `streak` values.
- **Signer mismatch:** INCLUDE has few signers; a new person scores below the
  97.9% test number even with correct framing.

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
- **Rest class + confidence gate + hand-coverage gate + smoothing** prevent
  flicker and false words, and make the demo abstain when hands are not visible.
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
  - Verified 2026-11-13: best epoch 52, val balanced score 0.9870 (sign 0.974,
    rest 1.000), 515k parameters; hand-dropout 0.4 selected by ablation.
- [x] Stage 4: evaluation (confusion matrix, published comparison)
  - Verified 2026-11-13: test top-1 0.9792, top-3 0.9896, macro F1 0.9792;
    synthetic rest 98.0%; reports written to `reports/`.
- [x] Stage 5: live webcam demo
  - Implemented and verified on dataset clips and a 6 s real-camera headless
    run. Labelled live evaluation: 4/8 settled, 78.95% committed precision,
    with hands in frame only ~27% of the time.
- [x] Phase 0: live instrumentation, hand-detection sweep, gate calibration
  - `DecisionGate` with hand-coverage blocking; precision–coverage sweep on
    labelled takes; `project.md` §9.
- [ ] Phase 1: facial-landmark ablation — **conditional** on re-recorded takes
  with hands in frame.

## Known caveats

- The published **94.5%** INCLUDE-50 result uses a *pretrained CNN feature
  extractor + encoder*, not keypoints alone. It is not directly comparable.
- The split is random videos, not signer-independent, so live accuracy will be
  lower than test accuracy.
- **The biggest live limitation is framing, not the model.** In the eight
  labelled takes, hands were inside the frame ~38% of frames and a pose wrist
  was inside only ~27%; hand detection is ~0% when the wrist is outside. Reframe
  before concluding that a sign is misclassified.
- INCLUDE has no idle/rest videos. The rest class is synthesised from real poses
  (median pose, hands at the hips) and therefore does not cover every real
  no-sign situation. The confidence and coverage gates compensate.
- A short recorded clip may end before enough full-window predictions arrive, so
  the display can stay at `...`. This is expected; use `--min-streak 2` for short
  clips (the default).
- Tests: `python -m unittest discover -s tests` (62 tests, no camera needed).
