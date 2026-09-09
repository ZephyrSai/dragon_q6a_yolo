# YOLO on Radxa Dragon Q6A (Qualcomm QCS6490) — NPU + Adreno GPU + CPU Diagnostic & Benchmark Suite (Linux + Android)

A full test suite for running [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO (YOLO11 and YOLO26) on the Radxa Dragon Q6A, a Qualcomm **QCS6490** "Dragonwing" board: the **Hexagon HTP NPU** (v68, ~12 TOPS) through the Qualcomm QNN stack, the **Adreno 643 GPU**, and the **Kryo 670 CPU**. On Linux it covers ONNX Runtime's QNN execution provider, Ultralytics' native QNN export, the QNN TensorFlow Lite delegate, Qualcomm's and Google's `benchmark_model`, NCNN on Vulkan (Mesa turnip), and CPU baselines. On **Android** it drives the same board over `adb`. It tells you which blocks actually work on your image and how fast they run.

Sibling suites with the same metrics and CSV schema: [ryzen_yolo](https://github.com/ZephyrSai/ryzen_yolo) (AMD), [openvino_yolo](https://github.com/ZephyrSai/openvino_yolo) (Intel), [rockchip_yolo](https://github.com/ZephyrSai/rockchip_yolo) (RK3588/RK3576).

## Hardware and stacks

| Block | Linux stack | Needs |
|---|---|---|
| NPU, Hexagon HTP v68 | `onnxruntime-qnn` (Linux aarch64 wheel; bundles QAIRT and the v68 skel) via ORT's QNN EP; Ultralytics `format=qnn name=68`; `libQnnTFLiteDelegate.so` | `/dev/fastrpc-cdsp`, CDSP firmware running, `libcdsprpc.so`, `ADSP_LIBRARY_PATH` |
| GPU, Adreno 643 | NCNN Vulkan on Mesa turnip (Radxa image default); QNN GPU backend and the TFLite GPU delegate on the proprietary Adreno OpenCL | `mesa-vulkan-drivers`; or `qcom-adreno-cl1` from Canonical's PPA |
| CPU, Kryo 670 (4× A78-class + 4× A55-class) | PyTorch, ONNX Runtime, NCNN, TFLite/XNNPACK | nothing |

The same silicon appears as QCM6490 and SM7325 (soc_id 497 / 475); the script accepts those too. Ultralytics' QNN export targets HTP architecture v68 with `name=68`, which is exactly this SoC.

## Quick start

TFLite files have to be produced on a PC (Ultralytics' LiteRT converter is x86-64 / macOS only). Everything else can be exported on the board, but the PC is faster:

```bash
# 1. PC (x86 Linux; macOS works for TFLite but not QNN)
./export_host.sh                     # add --quick for nano only
#    -> ~/yolo-q6a-export/q6a_models.tar.gz

# 2. Board (Radxa Ubuntu 24.04 image)
scp ~/yolo-q6a-export/q6a_models.tar.gz q6a:
./test_yolo_q6a.sh --models q6a_models.tar.gz

# 3. Android (from the PC; Radxa Android 15 image on eMMC/UFS)
./android/test_yolo_q6a_android.sh --models-dir ~/yolo-q6a-export
```

## What it does

### Linux, on the board (`test_yolo_q6a.sh`)

1. **System checks** — SoC id from `/sys/devices/soc0/soc_id`, device tree, kernel, `/dev/fastrpc-*` and their permissions, `remoteproc` state of ADSP and CDSP, DSP firmware files, `cdsprpcd`/`adsprpcd`, `libcdsprpc.so` (and the missing-unversioned-symlink trap), QAIRT libraries and skels on the system, `fastrpc_test -a v68` when installed, `dmesg` for the known "no reserved DMA memory for FASTRPC" signature, GPU driver (`msm` vs `msm_kgsl`), `vulkaninfo` (turnip), `clinfo` (Adreno blob or Mesa rusticl), Qualcomm's `benchmark_model`, CPU governor.
2. **Environment setup** — an isolated venv with Python ≥ 3.11 (needed by `onnxruntime-qnn`), installing Ultralytics, `onnxruntime` + `onnxruntime-qnn`, `ncnn`, `ai-edge-litert`. Downloads the test image, the six models, and the shared public test video.
3. **Runtime visibility** — registers the QNN EP plugin, points `ADSP_LIBRARY_PATH` at the wheel's bundled v68 skel, and compiles + runs a one-layer Conv on the HTP, GPU and QNN-CPU backends with CPU fallback disabled, so "works" means the accelerator really ran. Also probes the QNN TFLite delegate library, ncnn's Vulkan device, OpenCV OpenCL.
4. **Models** — unpacks the tarball, then fills in ONNX, NCNN and QNN exports on the board. The QNN export compiles a w8a16 HTP context binary (coco8 calibration) and can take minutes per model.
5. **Full benchmark**, every model × every backend that works, through Ultralytics end-to-end (preprocess + inference + NMS): `pytorch_cpu`, `qnn_npu_w8a16`, `onnxrt_cpu`, `ncnn_cpu`, `ncnn_vulkan`, `tflite_cpu_fp32`, `tflite_cpu_int8`. Cold-load, first-inference, steady-state avg/min/max/p95/stdev over 15 runs, real video FPS over up to 300 frames.
6. **Raw ONNX Runtime + QNN** — plain FP32 ONNX compiled by the QNN EP for HTP (FP16), GPU and QNN-CPU; the precompiled context on HTP under `burst`, `balanced` and `default` performance modes; ORT CPU EP baseline.
7. **Raw TFLite** — XNNPACK CPU, QNN delegate on HTP (INT8 and FP16 precision) and on GPU, plus `benchmark_model` (Qualcomm's from the Canonical PPA if installed, else Google's nightly Linux aarch64 binary) with the same delegates.
8. **NPU diagnosis** — one verdict with the fix: no fastrpc node, DSP firmware not running, `libcdsprpc` missing, node permissions, skel/runtime version mismatch (`0x80000600` / `14001`), or `onnxruntime-qnn` not installable.
9. **Summary** — pass/fail counts, comparison tables, env file with `ADSP_LIBRARY_PATH` and the fastest model/device.

### Android, from a host PC (`android/test_yolo_q6a_android.sh`)

1. Device identity from every SoC-revealing prop plus `soc_id`; presence of vendor QNN libs, skels, `/dev/fastrpc-cdsp`, NNAPI vendor HALs, Adreno OpenCL.
2. Google's prebuilt TFLite `benchmark_model` on every exported `.tflite`: CPU (XNNPACK), GPU delegate, NNAPI with `qti-dsp` / `qti-gpu` / `qti-default` accelerators, and, when you supply the QAIRT SDK's Android libs and v68 skel, the QNN delegate on HTP and GPU.
3. Pushes `qnn-net-run` when available and prints the command to benchmark a context binary directly.
4. Summary table + CSV.

## Requirements

**Board:** Radxa's Ubuntu 24.04 image (R2 or newer, which has the NPU runtime preinstalled), Armbian, or Canonical's Ubuntu for Qualcomm IoT. Python 3.11+ (Ubuntu 24.04 ships 3.12). For the NPU on the Radxa image: `sudo apt install fastrpc libcdsprpc1 fastrpc-test`. For GPU tests: `sudo apt install mesa-vulkan-drivers vulkan-tools mesa-opencl-icd clinfo`. For the QNN TFLite delegate: `qairt-libs` + `qairt-dsp-binaries` from `ppa:ubuntu-qcom-iot/qcom-ppa`, or a QAIRT SDK with `QNN_SDK_ROOT` set.

**PC (export):** Linux x86-64 with Python 3.11+ for the QNN export; macOS or Linux for TFLite.

**Android:** `adb`; Radxa's Android 15 image (eMMC/UFS, flashed in EDL mode). The unofficial AOSP/GloDroid image has no NPU stack and only adb over TCP.

## Output and backend labels

`benchmark_results.csv` uses the shared schema `backend,model,stage,metric,value`.

| Label | Meaning |
|---|---|
| `pytorch_cpu` | Ultralytics on `.pt`, end-to-end |
| `qnn_npu_w8a16` | Ultralytics on `<m>_qnn.onnx` (precompiled HTP v68 context, INT8 weights / 16-bit activations), end-to-end |
| `onnxrt_cpu`, `ncnn_cpu`, `ncnn_vulkan`, `tflite_cpu_{fp32,int8}` | Ultralytics on the other exports, end-to-end |
| `ort_cpu_raw` | ONNX Runtime CPU EP, raw tensors |
| `ort_qnn_htp_fp16_plainonnx`, `ort_qnn_gpu_plainonnx`, `ort_qnn_cpubackend` | QNN EP compiling the plain FP32 ONNX for each backend |
| `ort_qnn_htp_ctx_{burst,balanced,default}` | QNN EP running the precompiled context under each HTP performance mode |
| `tflite_raw_cpu`, `tflite_raw_qnn_htp`, `tflite_raw_qnn_gpu` | raw TFLite interpreter, XNNPACK / QNN delegate |
| `benchmark_model_{cpu,gpu,qnn_htp}` | Qualcomm's or Google's `benchmark_model` |
| `android_tflite_{cpu,gpu,nnapi_*,qnn_htp,qnn_gpu}` | `benchmark_model` on Android |

Only end-to-end rows are apples-to-apples with the other suites. Raw rows are device ceilings without letterboxing or NMS.

## NPU notes

- **The whole chain must line up:** kernel device tree with the fastrpc `memory-region`, CDSP firmware (`cdsp.mbn`) loaded and `remoteproc` in `running`, `/dev/fastrpc-cdsp` readable, `libcdsprpc.so` present, and a v68 skel (`libQnnHtpV68Skel.so`) from the **same QAIRT version** as the runtime, reachable through `ADSP_LIBRARY_PATH`. The `onnxruntime-qnn` wheel bundles a matching runtime + skel pair, which is why the suite prefers it; the script points `ADSP_LIBRARY_PATH` at the wheel automatically.
- **`0x80000600` / `Failed to create device: 14001`** almost always means a skel/runtime mismatch or missing DSP-side `libc++` symlinks, not broken hardware.
- **Only `libcdsprpc.so.1` installed?** QNN dlopens the unversioned name. Create the symlink the script suggests.
- **Ultralytics QNN export** runs on the board or a Linux PC (not macOS), needs no Qualcomm account, and enforces w8a16. YOLO11 and YOLO26 detection are both supported.
- **Qualcomm AI Hub** is the other route (`qai-hub-models`, target `QCS6490 (Proxy)`, TFLite w8a8 for YOLOv8/YOLO11). It needs an API token and its published YOLOv8-Det w8a8 number is about 4 ms on this NPU. Not automated here because of the token; the TFLite it produces drops straight into Steps 6/7 and the Android runner.
- **TFLite QNN delegate** needs QAIRT libraries on the system (`qairt-libs`), not just the wheel. Ultralytics' own TFLite backend on the board is CPU-only.

## GPU notes

- Radxa's image ships Mesa: `turnip` gives Vulkan 1.3 on the Adreno 643, so NCNN Vulkan works out of the box. `rusticl` OpenCL exists (`RUSTICL_ENABLE=freedreno`) but is slow.
- The QNN GPU backend and Qualcomm's `benchmark_model --use_gpu` link against the proprietary Adreno OpenCL (`cl_qcom_*` extensions). That comes from Canonical's PPA (`qcom-adreno-cl1`, and `qcom-adreno1` which replaces the Mesa desktop stack). Expect those rows to warn on the stock Radxa image.

## Android notes

- Only Radxa's official Android image carries the Qualcomm NPU stack. The NNAPI accelerator names on Qualcomm devices are `qti-dsp`, `qti-gpu`, `qti-default`; the script tries all three when a vendor HAL exists.
- The QNN delegate on Android needs `libQnnTFLiteDelegate.so`, `libQnnHtp*.so`, `libQnnSystem.so` from the QAIRT SDK's `lib/aarch64-android` and the skel from `lib/hexagon-v68/unsigned`. The SDK zip needs a Qualcomm ID to download; pass the paths with `--qnn-android-libs` and `--qnn-skel`.

## Files in this repo

- `export_host.sh` — PC-side export: ONNX, NCNN, QNN (HTP v68 context), TFLite fp32/int8; packs one tarball
- `test_yolo_q6a.sh` — board-side Linux diagnostic + benchmark driver
- `bench_harness.py` — Ultralytics end-to-end harness (shared with the other suites)
- `ort_qnn_bench.py` — raw ONNX Runtime + QNN EP benchmark (HTP / GPU / CPU backends, performance modes)
- `tflite_bench.py` — raw TFLite benchmark with external delegate + options (QNN delegate)
- `android/test_yolo_q6a_android.sh` — host-side adb driver for Android

## Caveats

- Verified so far: the harness scripts on a non-Qualcomm machine (CPU paths, TFLite FP32/uint8 handling, ONNX Runtime CPU). Every QNN, FastRPC, Adreno and Android path needs the real board; the driver scripts degrade to warnings rather than aborting.
- The QNN export calibrates on `coco8`. Calibrate on a few hundred representative images for real deployments.
- Raw rows exclude letterboxing and NMS.
