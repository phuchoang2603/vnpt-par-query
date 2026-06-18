# PP-Human Attribute Recognition on Qualcomm AI Box QCS6490

Implementation plan for running the full PP-Human attribute pipeline — **detection, tracking, and attribute analysis** — on a Qualcomm AI Box (QCS6490) over an **RTSP video stream**.

**Target stack:** Qualcomm AI Engine Direct (**QNN**) + pybind11, developed on **Ubuntu 24.04 VM**, deployed on **AI Box Ubuntu 24.04** with **HTP (DSP)** as primary runtime.

Related:
- [PP-Human Attribute](PaddleDetection/deploy/pipeline/docs/tutorials/pphuman_attribute_en.md)
- [PP-Human MOT](PaddleDetection/deploy/pipeline/docs/tutorials/pphuman_mot_en.md)
- [SNPE YOLOv7 reference](../snpe-yolov7-inference/) (legacy pattern for pybind11)
- [STT 16: Context-Based Person Search](Crowd%20Analytics%20and%20Spatial%20Intelligence.md#4-stt-16-context-based-person-search)
- [AI Hub Workbench](https://aihub.qualcomm.com/get-started#workbench)
- [QNN HTP tutorial (QCS6490)](https://docs.qualcomm.com/nav/home/qnn_tutorial_linux_host_linux_target_htp.html)

---

## 0. Migration summary: SNPE → QNN

If you started from the SNPE plan, these are the deltas:

| Area | SNPE (old) | QNN (target) |
|------|------------|--------------|
| SDK | Snapdragon Neural Processing Engine | **Qualcomm AI Engine Direct (QNN / QAIRT)** |
| Model artifact | `.dlc` container | **`.so` model lib** + **`.serialized.bin` context** (HTP) |
| Primary runtime | GPU (DSP failed on Ubuntu 20) | **HTP INT8** (Ubuntu 24 + v68 Hexagon libs) |
| Smoke test | `snpe-net-run --use_gpu` | `qnn-net-run --backend libQnnHtp.so --retrieve_context …` |
| pybind11 module | `snpe_runtime` (SNPE C++ API) | **`qnn_runtime`** (QNN C API) |
| Conversion | AI Hub → SNPE DLC | **AI Hub → QNN** *or* **QNN SDK on Ubuntu 24 VM** |
| Dev machine | Mac (paddle2onnx only) | **Ubuntu 24 VM** (full QNN host toolchain) |
| Qualcomm direction | Maintenance mode | **Recommended for new deployments** |

**What stays the same:** RTSP 3-thread pipeline, OC-SORT tracking, PPLCNet attribute postprocess, STT 16 metadata schema, PP-YOLOE-s + PPLCNet models, pybind11 pattern.

---

## 1. Scope

| In scope | Out of scope (later) |
|----------|----------------------|
| RTSP video ingest | Single-image batch mode |
| PP-YOLOE detection + OC-SORT tracking | Multi-camera MTMCT / ReID |
| PPLCNet attribute recognition | MQTT fan-out |
| pybind11 **QNN** runtime | PaddlePaddle on device |
| Metadata events for STT 16 DB export | Full PostGIS schema (stub only) |

```
RTSP frame → MOT (detect + track) → crop tracked persons → ATTR → metadata emit
```

---

## 2. Target hardware & runtime (Ubuntu 24)

### 2.1 Hardware spec

| Component | Spec | Role |
|-----------|------|------|
| Kryo 670 CPU | 2× Gold+ @ 2.7 GHz, 3× Gold @ 2.4 GHz, 4× Silver @ 1.9 GHz | OC-SORT, preprocess/postprocess, RTSP, DB writer |
| Adreno 643 GPU | FP16/FP32 | **Fallback** runtime |
| Hexagon DSP + HTP | v68, dual HVX, Tensor Accelerator | **Primary runtime** (12 TOPS) |
| Spectra 570L ISP | — | Camera / RTSP |

### 2.2 Runtime recommendation (post Ubuntu 24 upgrade)

On Ubuntu 20, `snpe-platform-validator` reported DSP prerequisites absent. **Ubuntu 24 BSP** should ship matching Hexagon firmware, FastRPC libs, and `libQnnHtpV68Skel.so` paths — re-validate after upgrade.

| Runtime | When to use | Precision | QNN backend |
|---------|-------------|-----------|-------------|
| **HTP (DSP)** | **Production default** | INT8 quantized | `libQnnHtp.so` + `.serialized.bin` |
| GPU | Fallback / debug | FP16 | `libQnnGpu.so` |
| CPU | Parity only | FP32 | `libQnnCpu.so` |

Runtime priority: **HTP → GPU → CPU**.

**First action after OS upgrade:**

```bash
# Re-validate on AI Box (QNN SDK ships platform tools)
# Confirm hexagon-v68 skel libs present under $QNN_SDK_ROOT/lib/hexagon-v68/
qnn-net-run --backend libQnnHtp.so ...   # smoke test with a known model
```

Set on AI Box (typical for HTP):

```bash
export QNN_SDK_ROOT=/opt/qcom/aistack/qnn/<version>
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/aarch64-linux-clang:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=$QNN_SDK_ROOT/lib/hexagon-v68/unsigned
```

### 2.3 Production config

```yaml
MOT:
  backend: HTP
  model_lib: models/aarch64-linux-clang/libppyoloe_mot_quantized.so
  context_bin: models/ppyoloe_mot_htp_v68.serialized.bin
  htp_arch: v68          # QCS6490

ATTR:
  backend: HTP
  model_lib: models/aarch64-linux-clang/libpplcnet_attr_quantized.so
  context_bin: models/pplcnet_attr_htp_v68.serialized.bin
  batch_size: 8
```

---

## 3. Model selection

Unchanged — edge-appropriate fast variants:

| Role | Model | Why |
|------|-------|-----|
| Detection + MOT | **PP-YOLOE-s** (`mot_ppyoloe_s_36e_pipeline`) | Speed; single-class person |
| Attribute | **PP-LCNet x1.0** (`PPLCNet_x1_0_person_attribute_945_infer`) | Fastest; mA 94.5 |

HTP INT8 unlocks higher sustained FPS — `skip_frame_num` can drop from 2 → 0–1 after profiling.

---

## 4. Architecture

### 4.1 Stack

```
┌─────────────────────────────────────────────────────────────┐
│  Python pipeline (Ubuntu 24 VM dev → AI Box deploy)         │
│  ├─ RTSP capture thread                                     │
│  ├─ Inference thread (det QNN HTP → tracker → attr QNN HTP) │
│  └─ Metadata writer thread (STT 16)                         │
├─────────────────────────────────────────────────────────────┤
│  pybind11 module: qnn_runtime                               │
│  QNN C API: graph compose → execute → output tensors        │
├─────────────────────────────────────────────────────────────┤
│  AI Box Ubuntu 24                                           │
│  libQnnHtp.so │ libQnnHtpV68Stub.so │ hexagon-v68 skel      │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Pipeline dataflow

```mermaid
flowchart TB
    subgraph capture ["① Capture thread"]
        direction LR
        RTSP["RTSP stream"] --> READ["Read frame"] --> FQ[("frame_queue<br/>max 3 frames")]
    end

    subgraph infer ["② Inference thread"]
        direction LR
        DET["Detect<br/>PP-YOLOE-s · HTP"] --> TRK["Track<br/>OC-SORT"] --> CROP["Crop<br/>persons"] --> ATTR["Attributes<br/>PPLCNet · HTP"] --> LABELS["Decode<br/>26 labels"]
    end

    subgraph write ["③ Writer thread"]
        direction LR
        EQ[("event_queue")] --> WR["Batch writer"] --> DB[("PostGIS<br/>STT 16")]
    end

    FQ --> DET
    LABELS --> EQ
```

### 4.3 Threading

Unchanged — **3-thread in-process queue** (not MQTT for v1):

1. Capture → `frame_queue`
2. Inference → `event_queue`
3. Writer → PostGIS batch insert

### 4.4 STT 16 metadata

Same `PersonEvent` JSON schema as before (track_id, bbox, centroid, attributes, timestamp). No raw frames stored.

---

## 5. Development environments

### 5.1 Two-machine setup

| Machine | OS | Role |
|---------|-----|------|
| **Dev VM** | Ubuntu 24.04 on Mac (UTM/Parallels/VMware) | Repo, paddle2onnx, QNN SDK **host** tools, pybind11 **cross-compile**, unit tests |
| **AI Box** | Ubuntu 24.04 (upgraded) | QNN SDK **target** libs, HTP inference, RTSP production |

Mac host is no longer the conversion machine — the Ubuntu 24 VM replaces it.

### 5.2 Dev VM setup (one-time)

1. Install Ubuntu 24.04 VM with ≥ 8 GB RAM, 40 GB disk
2. Install Qualcomm **QNN SDK** (QAIRT) for Linux x86_64 — same major version as AI Box
3. `source $QNN_SDK_ROOT/bin/envsetup.sh`
4. Python venv: `numpy`, `opencv-python`, `pyyaml`, `paddle2onnx`, `paddlepaddle` (for export only)
5. Optional: `qai-hub` client if using AI Hub for compile jobs from VM
6. Cross-compile toolchain for `aarch64-linux-gnu` (pybind11 `.so` for AI Box)

Clone repo into VM; edit, build bindings, run parity tests against QNN CPU backend on VM before deploying to box.

### 5.3 AI Box setup (after Ubuntu 24 upgrade)

1. Install matching QNN SDK **target** package (or use vendor BSP bundle)
2. Re-run HTP validation with `qnn-net-run` + Mobilenet demo
3. Deploy model artifacts + `qnn_runtime` pybind11 module
4. Configure `ADSP_LIBRARY_PATH` for hexagon-v68

**SDK version parity:** host (VM) and target (AI Box) QNN SDK versions must match for context binaries.

---

## 6. Model conversion workflow

Two valid paths — pick one per model, not both.

### 6.1 Path A — AI Hub Workbench (recommended to start)

Same cloud flow as SNPE plan, but change **target runtime to QNN** (AI Hub labels Compute/IoT devices as QNN).

```mermaid
flowchart LR
    A[Paddle models] --> B[paddle2onnx on Ubuntu 24 VM]
    B --> C[AI Hub Workbench]
    C --> D[Compile QCS6490 + QNN + HTP]
    D --> E[Quantize + profile]
    E --> F[Download .so + context binary]
    F --> G[Deploy to AI Box]
```

| Step | Where |
|------|-------|
| Paddle → ONNX | Ubuntu 24 VM |
| Compile / quantize / context-gen | AI Hub Workbench |
| Profile | Cloud-hosted QCS6490 |
| Deploy | AI Box |

**AI Hub job settings (per model):**

- Device: **QCS6490**
- Runtime: **QNN** (not SNPE, not TFLite)
- Backend: **HTP**
- Precision: **INT8** (required for HTP)

Download artifacts — expect **QNN model library** (`.so`) and/or **serialized context** (`.bin`), not DLC. Record graph name, input/output tensor names from job report.

Calibration data: 50–100 RTSP frames (det) + pedestrian crops (attr), preprocessed identically to runtime.

### 6.2 Path B — QNN SDK on Ubuntu 24 VM (full local control)

Use when you need reproducible CI, custom quant, or AI Hub rejects an op.

**Per-model pipeline:**

| Step | Tool | Output |
|------|------|--------|
| 1. Paddle → ONNX | `paddle2onnx` | `*.onnx` |
| 2. ONNX → QNN | `qnn-onnx-converter` + `--input_list` | `model.cpp` + weights |
| 3. Build for target | `qnn-model-lib-generator` | `lib<model>_quantized.so` (aarch64) |
| 4. HTP context | `qnn-context-binary-generator` + `htp_arch: v68` config | `*.serialized.bin` |
| 5. Smoke test | `qnn-net-run` on VM (CPU) then AI Box (HTP) | — |

**QCS6490 HTP backend config** (mandatory for context generation):

```json
{
  "graphs": [{ "graph_names": ["<your_graph_name>"], "vtcm_mb": 2 }],
  "devices": [{ "htp_arch": "v68" }]
}
```

Reference: [QNN HTP tutorial for QCS6490](https://docs.qualcomm.com/nav/home/qnn_tutorial_linux_host_linux_target_htp.html).

**Deploy bundle per model:**

- `lib<model>_quantized.so` (aarch64)
- `<model>_htp_v68.serialized.bin`
- `libQnnHtpV68Stub.so`, `libQnnHtpPrepare.so` (from SDK)
- `libQnnHtpV68Skel.so` (hexagon-v68 unsigned)

No conversion scripts live in the plan repo — document commands in project README when implementing.

### 6.3 Which path when?

| Situation | Path |
|-----------|------|
| First bring-up, minimal SDK friction | **AI Hub** (QNN + HTP) |
| Repeatable builds, CI, custom quant | **QNN SDK on VM** |
| AI Hub rejects PP-YOLOE op | **QNN SDK on VM** with manual graph surgery / onnx-simplifier |

---

## 7. Project layout

```
qnn-pphuman-pipeline/
├── bindings/
│   ├── qnn_runtime.cpp        # pybind11 over QNN C API
│   ├── qnn_runtime.hpp
│   └── CMakeLists.txt         # cross-compile aarch64-linux-gnu
├── config/
│   ├── infer_cfg_qnn.yml
│   ├── htp_backend_v68.json   # context-binary-generator config
│   └── tracker_config.yml
├── models/                    # gitignored
│   ├── aarch64-linux-clang/
│   │   ├── libppyoloe_mot_quantized.so
│   │   └── libpplcnet_attr_quantized.so
│   ├── ppyoloe_mot_htp_v68.serialized.bin
│   └── pplcnet_attr_htp_v68.serialized.bin
├── src/
│   ├── qnn_engine.py          # Python wrapper (replaces snpe_engine.py)
│   ├── rtsp_capture.py
│   ├── det_engine.py
│   ├── tracker.py
│   ├── attr_engine.py
│   ├── crop_utils.py
│   ├── metadata.py
│   ├── db_writer.py
│   ├── pipeline.py
│   └── visualize.py
├── tools/
│   ├── generate_calib_list.py
│   └── validate_parity.py
├── requirements.txt
└── README.md
```

**pybind11 `qnn_runtime` API (sketch):**

```python
engine = QNNEngine(
    backend="HTP",                          # HTP | GPU | CPU
    model_lib="models/.../libmodel.so",
    context_bin="models/model_htp_v68.serialized.bin",  # HTP only
)
output = engine.execute(input_chw_float32)  # np.ndarray → np.ndarray
```

Internally: load backend → create device → load context binary (HTP) or compose graph (GPU/CPU) → `QnnGraph_execute`.

---

## 8. Config (`config/infer_cfg_qnn.yml`)

```yaml
rtsp_url: "rtsp://user:pass@camera-ip:554/stream1"
camera_id: "cam_01"

crop_thresh: 0.5
attr_thresh: 0.5
warmup_frame: 10
skip_frame_num: 1          # can reduce to 0 with HTP

frame_queue_size: 3
event_flush_interval_s: 5

MOT:
  enable: true
  backend: HTP
  model_lib: models/aarch64-linux-clang/libppyoloe_mot_quantized.so
  context_bin: models/ppyoloe_mot_htp_v68.serialized.bin
  input_size: [640, 640]
  input_name: x
  score_thresh: 0.5
  tracker_type: OCSORTTracker

ATTR:
  enable: true
  backend: HTP
  model_lib: models/aarch64-linux-clang/libpplcnet_attr_quantized.so
  context_bin: models/pplcnet_attr_htp_v68.serialized.bin
  input_size: [192, 256]
  batch_size: 8

DB:
  enable: false
```

---

## 9. Code to port

| Module | Source | QNN change |
|--------|--------|------------|
| Runtime binding | `snpe-yolov7-inference/SNPEPipeline.cpp` | Rewrite for **QNN C API** (different lifecycle) |
| Preprocess / postprocess / tracker / crop / attr decode | PaddleDetection deploy | **Unchanged** |
| Pipeline orchestration | `pipeline.py` | Swap `SNPEEngine` → `QNNEngine` |

Python preprocessing and OC-SORT are runtime-agnostic — only the binding layer changes.

---

## 10. Implementation phases

### Phase 0 — Environment
- [ ] Ubuntu 24 VM: QNN SDK host + dev toolchain
- [ ] AI Box: upgrade to Ubuntu 24, install matching QNN SDK target
- [ ] HTP smoke test on AI Box (`qnn-net-run` + v68 libs)
- [ ] RTSP connectivity test

### Phase 1 — Model conversion
- [ ] Download Paddle models on VM
- [ ] `paddle2onnx` → validate in Netron
- [ ] **AI Hub:** compile both models for QCS6490 / QNN / HTP / INT8
- [ ] **Or QNN SDK:** converter → quantize → context-binary-generator (v68)
- [ ] Profile on AI Hub or `qnn-net-run` on device

### Phase 2 — pybind11 QNN runtime
- [ ] Implement `qnn_runtime` (HTP context load + execute)
- [ ] Cross-compile for aarch64; deploy to AI Box
- [ ] Python smoke test

### Phase 3 — Detection + tracking
- [ ] `det_engine.py` + OC-SORT
- [ ] Parity vs Paddle CPU

### Phase 4 — Attribute
- [ ] Batched HTP inference + `postprocess()`
- [ ] Validate INT8 accuracy on attribute logits

### Phase 5 — RTSP pipeline
- [ ] 3-thread pipeline end-to-end
- [ ] Tune FPS with HTP profiling

### Phase 6 — STT 16 metadata export
- [ ] `PersonEvent` → PostGIS writer

---

## 11. Performance targets (HTP INT8, 1080p RTSP)

| Component | Target | Notes |
|-----------|--------|-------|
| Detection | < 25 ms/frame | HTP vs ~50 ms GPU-only |
| Tracking | < 5 ms | CPU |
| Attribute | < 3 ms/person batched | HTP |
| End-to-end | **15–25 FPS** | 1 cam, 3–5 persons; re-measure |

---

## 12. Risk register

| Risk | Mitigation |
|------|------------|
| DSP still fails on Ubuntu 24 | Verify BSP with vendor; check `ADSP_LIBRARY_PATH`; GPU fallback |
| QNN SDK version mismatch VM ↔ box | Pin same QAIRT version in both environments |
| HTP INT8 hurts attribute mA | More calibration crops; FP16 GPU for ATTR only |
| AI Hub outputs wrong artifact type | Select **QNN** runtime explicitly; not SNPE DLC |
| PP-YOLOE unsupported op | QNN SDK local path + onnx-simplifier |
| Cross-compile binding errors | Build natively on AI Box as fallback |

---

## 13. Decision log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Inference stack | **QNN** (not SNPE) | Qualcomm recommended; better HTP path |
| Dev environment | **Ubuntu 24 VM on Mac** | Full QNN host SDK; cross-compile |
| AI Box OS | **Ubuntu 24** | DSP/HTP prerequisites |
| Primary runtime | **HTP INT8 v68** | 12 TOPS NPU; QCS6490 |
| Fallback runtime | GPU FP16 | If HTP op unsupported |
| Models | PP-YOLOE-s + PPLCNet | Unchanged |
| Binding | pybind11 | Unchanged pattern, new API |
| Conversion | AI Hub (QNN) **or** QNN SDK on VM | AI Hub first; SDK for control |
| Threading | 3-thread queue | Unchanged; STT 16 ready |
| MQTT | Deferred | Unchanged |

---

## 14. Next actions

1. Upgrade AI Box to Ubuntu 24; install QNN SDK; confirm HTP with `qnn-net-run`.
2. Set up Ubuntu 24 VM with matching QNN SDK host tools.
3. `paddle2onnx` both models on VM.
4. AI Hub Workbench: compile for **QCS6490 / QNN / HTP** → download artifacts.
5. Scaffold `qnn-pphuman-pipeline/` with `qnn_runtime` pybind11 binding.

---

## Appendix A — SNPE interim path (optional)

If you need something running **before** QNN migration completes, the previous SNPE + GPU plan still works on Ubuntu 24:

- AI Hub → SNPE DLC → `snpe-net-run --use_gpu` → `snpe_runtime` pybind11
- Expect lower FPS; swap to QNN HTP when context binaries are ready
- **Do not invest in SNPE HTP graph-prepare** — go straight to QNN for DSP

Once QNN pipeline is validated, retire SNPE artifacts and bindings.
