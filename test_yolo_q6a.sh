#!/usr/bin/env bash
#
# test_yolo_q6a.sh — run ON THE BOARD (Radxa Dragon Q6A, Qualcomm QCS6490,
# Linux: Radxa Ubuntu 24.04 image, Armbian, or Canonical's Ubuntu for
# Qualcomm IoT)
#
# Full diagnostic + benchmark suite for Ultralytics YOLO (YOLO11 + YOLO26)
# on the QCS6490: Hexagon HTP NPU (v68, ~12 TOPS) through the QNN stack
# (ONNX Runtime QNN EP, Ultralytics QNN export, TFLite QNN delegate,
# Qualcomm's benchmark_model), the Adreno 643 GPU (NCNN Vulkan on turnip,
# QNN GPU backend on the proprietary OpenCL stack), and the Kryo 670 CPU
# (PyTorch, ONNX Runtime, NCNN, TFLite/XNNPACK).
#
#   Block          Stack on Linux                                   Needs
#   ------------   ----------------------------------------------   -----------------------------------------
#   NPU (HTP v68)  onnxruntime-qnn wheel (bundles QAIRT + v68 skel)  /dev/fastrpc-cdsp, cdsp firmware running,
#                  Ultralytics format=qnn name=68                    libcdsprpc.so, ADSP_LIBRARY_PATH
#                  libQnnTFLiteDelegate.so (QAIRT SDK / qairt-libs)
#   GPU (A643)     NCNN Vulkan (Mesa turnip, Radxa image default)    mesa-vulkan-drivers
#                  QNN GPU backend / TFLite GPU delegate             proprietary qcom-adreno-cl1 (Canonical PPA)
#   CPU            PyTorch, ORT, NCNN, TFLite XNNPACK                nothing
#
# Model coverage: YOLO11 + YOLO26, nano/small/medium (--quick = nano only).
# TFLite files must come from a PC (Ultralytics' LiteRT converter is
# x86/macOS only): ./export_host.sh on the PC -> --models q6a_models.tar.gz.
# ONNX, NCNN and QNN exports are done on the board when missing.
#
# Tests, in order:
#   0. System checks: SoC id, device tree, kernel, fastrpc nodes + perms,
#      remoteproc adsp/cdsp state, DSP firmware, cdsprpcd/adsprpcd, libcdsprpc,
#      QAIRT libs/skels, fastrpc_test, GPU driver/Vulkan/OpenCL, governors
#   1. Python env (>= 3.11 for onnxruntime-qnn) + assets/models
#   2. Runtime visibility: ORT QNN EP registration + a trivial Conv session on
#      HTP / GPU / QNN-CPU, TFLite runtime + QNN delegate lib, ncnn Vulkan, OpenCV OpenCL
#   3. Models: tarball or on-board export (ONNX, NCNN, QNN-HTP context)
#   4. FULL BENCHMARK (Ultralytics end-to-end, with NMS): pytorch_cpu, qnn_npu,
#      onnxrt_cpu, ncnn_cpu, ncnn_vulkan, tflite_cpu_{fp32,int8}
#   5. Raw ORT-QNN: plain FP32 ONNX on HTP (fp16) / GPU / QNN-CPU, and the
#      precompiled context on HTP under burst / balanced / default perf modes
#   6. Raw TFLite: XNNPACK, QNN delegate HTP (int8 + fp16), QNN delegate GPU;
#      Qualcomm/Google benchmark_model with the same delegates
#   7. NPU diagnosis with the exact fix
#   8. Summary tables + env file
#
# Usage:
#   ./test_yolo_q6a.sh --models q6a_models.tar.gz [--quick] [--skip-install] [--python python3.12]
#
# Everything lives under ~/yolo-q6a-test.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$HOME/yolo-q6a-test"; VENV_DIR="$WORKDIR/venv"; LOGDIR="$WORKDIR/logs"
RESULTS_FILE="$WORKDIR/results_summary.txt"; BENCH_CSV="$WORKDIR/benchmark_results.csv"
SKIP_INSTALL=0; QUICK_MODE=0; MODELS_IN=""; PYBIN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-install) SKIP_INSTALL=1 ;; --quick) QUICK_MODE=1 ;;
    --models) MODELS_IN="$2"; shift ;; --python) PYBIN="$2"; shift ;;
    -h|--help) sed -n '2,50p' "$0"; exit 0 ;;
  esac; shift
done
mkdir -p "$WORKDIR" "$LOGDIR"; : > "$RESULTS_FILE"; echo "backend,model,stage,metric,value_ms_or_fps" > "$BENCH_CSV"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
section() { echo -e "\n${BLUE}==================================================================${NC}\n${BLUE}  $1${NC}\n${BLUE}==================================================================${NC}"; }
pass() { echo -e "${GREEN}[PASS]${NC} $1"; echo "[PASS] $1" >> "$RESULTS_FILE"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; echo "[FAIL] $1" >> "$RESULTS_FILE"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; echo "[WARN] $1" >> "$RESULTS_FILE"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }

# ==================================================================
section "STEP 0: System / driver sanity checks"
# ==================================================================
echo "Kernel: $(uname -r)  arch: $(uname -m)" | tee "$LOGDIR/kernel.log"
[ "$(uname -m)" = "aarch64" ] || fail "Not aarch64 — run this on the Dragon Q6A itself"
SOC_ID=$(cat /sys/devices/soc0/soc_id 2>/dev/null); COMPAT=$(tr '\0' '\n' < /proc/device-tree/compatible 2>/dev/null); BOARD=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)
echo "board=$BOARD soc_id=$SOC_ID compat=$(echo "$COMPAT" | tr '\n' ' ')" | tee "$LOGDIR/board.log"
case "$SOC_ID" in
  498) pass "SoC: QCS6490 (soc_id 498) — Hexagon v68 HTP, Adreno 643 — board: ${BOARD:-?}" ;;
  497|475) pass "SoC: QCM6490/SM7325 (soc_id $SOC_ID) — same Kodiak silicon as QCS6490" ;;
  *) if echo "$COMPAT" | grep -qE "qcom,qc[ms]6490|qcs6490|sm7325"; then pass "SoC: QCS6490-family from device tree ($(echo "$COMPAT" | tail -1))"; else warn "Could not confirm QCS6490 (soc_id='${SOC_ID:-none}', compat='$(echo "$COMPAT" | tail -1)') — continuing with v68 assumptions"; fi ;;
esac
echo "$COMPAT" | grep -q "radxa,dragon-q6a" && pass "Device tree: radxa,dragon-q6a" || info "device-tree compatible does not name dragon-q6a (vendor kernel or different board)"

echo -e "\nFastRPC / Hexagon NPU plumbing:"
NPU_PLUMBING=1
if ls /dev/fastrpc-cdsp >/dev/null 2>&1; then
  pass "/dev/fastrpc-cdsp present ($(ls /dev/fastrpc-* | tr '\n' ' '))"
  { [ -r /dev/fastrpc-cdsp ] && [ -w /dev/fastrpc-cdsp ]; } && pass "/dev/fastrpc-cdsp accessible by $USER" || { warn "/dev/fastrpc-cdsp NOT accessible by $USER — udev rule: KERNEL==\"fastrpc-*\", MODE=\"0666\" (plus dma_heap system) or add user to the fastrpc group"; NPU_PLUMBING=0; }
else
  fail "/dev/fastrpc-cdsp missing — fastrpc driver not bound (kernel DT lacks fastrpc/memory-region, CONFIG_QCOM_FASTRPC off, or cdsp remoteproc failed)"; NPU_PLUMBING=0
fi
lsmod | grep -q fastrpc && info "fastrpc module loaded" || { grep -q "CONFIG_QCOM_FASTRPC=y" /boot/config-$(uname -r) 2>/dev/null && info "fastrpc built-in" || warn "fastrpc module not loaded/built-in"; }
for r in /sys/class/remoteproc/remoteproc*; do
  [ -e "$r/name" ] || continue
  n=$(cat "$r/name" 2>/dev/null); s=$(cat "$r/state" 2>/dev/null)
  case "$n" in *cdsp*|*adsp*) [ "$s" = "running" ] && pass "remoteproc $n: running" || { warn "remoteproc $n: $s (firmware missing/corrupt? check dmesg)"; [[ "$n" == *cdsp* ]] && NPU_PLUMBING=0; } ;; esac
done
ls /lib/firmware/qcom/qcs6490/cdsp.mbn /lib/firmware/updates/qcom/qcs6490/cdsp.mbn /lib/firmware/qcom/qcs6490/*/*/cdsp.mbn >/dev/null 2>&1 && pass "CDSP firmware present" || warn "No cdsp.mbn under /lib/firmware/qcom/qcs6490 — install radxa-firmware-qcs6490 / linux-firmware-dragonwing"
for svc in cdsprpcd adsprpcd; do systemctl is-active --quiet "$svc" 2>/dev/null && info "$svc.service active" || info "$svc.service not active (optional: Canonical qcom-fastrpc1 daemons)"; done
LIBCDSP=$(ldconfig -p 2>/dev/null | awk '/libcdsprpc.so/{print $NF; exit}')
if [ -n "$LIBCDSP" ]; then
  pass "libcdsprpc: $LIBCDSP"
  ldconfig -p | grep -q "libcdsprpc.so " || warn "only libcdsprpc.so.1 present — QNN libs dlopen 'libcdsprpc.so'; fix: sudo ln -s $(basename "$LIBCDSP") $(dirname "$LIBCDSP")/libcdsprpc.so"
else
  warn "libcdsprpc.so missing — Radxa: sudo apt install fastrpc libcdsprpc1 ; Canonical PPA: qcom-fastrpc1"; NPU_PLUMBING=0
fi
SKEL_SYS=""; for d in /usr/lib/rfsa/adsp /usr/lib/dsp/cdsp /usr/lib/rfsa/adsp/cdsp /dsp /usr/lib; do [ -f "$d/libQnnHtpV68Skel.so" ] && { SKEL_SYS="$d"; break; }; done
[ -n "$SKEL_SYS" ] && pass "System HTP v68 skel: $SKEL_SYS/libQnnHtpV68Skel.so" || info "no system libQnnHtpV68Skel.so (the onnxruntime-qnn wheel bundles one; TFLite delegate needs QAIRT libs)"
QNN_SYS_LIB=""; for d in /usr/lib /usr/lib/aarch64-linux-gnu /opt/qcom/aistack/qairt/*/lib/aarch64-ubuntu-gcc9.4 /opt/qcom/aistack/qairt/*/lib/aarch64-oe-linux-gcc11.2 "${QNN_SDK_ROOT:-/nonexistent}"/lib/aarch64-ubuntu-gcc9.4 "${QNN_SDK_ROOT:-/nonexistent}"/lib/aarch64-oe-linux-gcc11.2; do [ -f "$d/libQnnHtp.so" ] && { QNN_SYS_LIB="$d"; break; }; done
[ -n "$QNN_SYS_LIB" ] && pass "QAIRT libs: $QNN_SYS_LIB (libQnnHtp.so$( [ -f "$QNN_SYS_LIB/libQnnTFLiteDelegate.so" ] && echo ', libQnnTFLiteDelegate.so'))" || info "no QAIRT SDK libs on the system (qairt-libs from ppa:ubuntu-qcom-iot/qcom-ppa, or QNN_SDK_ROOT) — TFLite QNN delegate rows will be skipped; ORT-QNN uses its own wheel"
if command -v fastrpc_test >/dev/null 2>&1; then
  fastrpc_test -a v68 > "$LOGDIR/fastrpc_test.log" 2>&1
  grep -q "All applicable tests PASSED\|Failed: 0" "$LOGDIR/fastrpc_test.log" && pass "fastrpc_test -a v68: PASSED" || { warn "fastrpc_test failed — see $LOGDIR/fastrpc_test.log"; NPU_PLUMBING=0; }
else
  info "fastrpc_test not installed (Radxa: sudo apt install fastrpc-test) — skipping"
fi
(dmesg 2>/dev/null || sudo -n dmesg 2>/dev/null) | grep -iE "fastrpc|cdsp|adsp|remoteproc" | tail -15 > "$LOGDIR/dmesg_npu.log"
grep -qi "no reserved DMA memory for FASTRPC" "$LOGDIR/dmesg_npu.log" && { fail "dmesg: 'no reserved DMA memory for FASTRPC' — kernel DT is missing the fastrpc memory-region (known on some 6.18 test kernels); update to Radxa R2+ kernel/overlay"; NPU_PLUMBING=0; }

echo -e "\nGPU (Adreno 643):"
GPU_KMOD="none"; for m in msm msm_kgsl; do lsmod | grep -qw "$m" && GPU_KMOD="$m"; done; [ -d /sys/module/msm ] && GPU_KMOD="msm"
[ "$GPU_KMOD" != "none" ] && pass "GPU kernel driver: $GPU_KMOD $([ "$GPU_KMOD" = msm_kgsl ] && echo '(proprietary Adreno stack)' || echo '(mainline drm/msm)')" || warn "No msm/msm_kgsl GPU driver"
ls /dev/dri/renderD* >/dev/null 2>&1 && pass "render nodes: $(ls /dev/dri/renderD* | tr '\n' ' ')" || warn "no /dev/dri/renderD*"
groups | grep -qE '\b(render|video)\b' && pass "User in render/video group" || warn "User not in render/video group (sudo usermod -aG render,video \$USER)"
VULKAN_OK=0; if command -v vulkaninfo >/dev/null 2>&1; then vulkaninfo --summary > "$LOGDIR/vulkaninfo.log" 2>&1; if grep -qiE "deviceName.*(Adreno|643)" "$LOGDIR/vulkaninfo.log"; then pass "Vulkan: $(grep -m1 deviceName "$LOGDIR/vulkaninfo.log" | sed 's/.*= *//') ($(grep -m1 driverName "$LOGDIR/vulkaninfo.log" | sed 's/.*= *//'))"; VULKAN_OK=1; else warn "vulkaninfo shows no Adreno (sudo apt install mesa-vulkan-drivers)"; fi; else info "vulkaninfo not installed (sudo apt install vulkan-tools)"; fi
OPENCL_OK=0; OPENCL_KIND="none"
if command -v clinfo >/dev/null 2>&1; then
  clinfo > "$LOGDIR/clinfo.log" 2>&1
  if grep -qiE "Device Name.*(Adreno|QUALCOMM)" "$LOGDIR/clinfo.log"; then OPENCL_OK=1; OPENCL_KIND="adreno"; pass "OpenCL: proprietary Adreno ($(grep -m1 -iE 'Device Version' "$LOGDIR/clinfo.log" | sed 's/.*Version[[:space:]]*//'))"
  else RUSTICL_ENABLE=freedreno clinfo > "$LOGDIR/clinfo_rusticl.log" 2>&1; if grep -qiE "Device Name.*FD6" "$LOGDIR/clinfo_rusticl.log"; then OPENCL_OK=1; OPENCL_KIND="rusticl"; pass "OpenCL: Mesa rusticl/freedreno (export RUSTICL_ENABLE=freedreno) — slow, and QNN GPU backend needs the Adreno blob"; export RUSTICL_ENABLE=freedreno; else info "no OpenCL device (mesa-opencl-icd + RUSTICL_ENABLE=freedreno, or qcom-adreno-cl1 from the Canonical PPA)"; fi; fi
else info "clinfo not installed"; fi
command -v benchmark_model >/dev/null 2>&1 && pass "Qualcomm benchmark_model on PATH ($(command -v benchmark_model))" || info "no /usr/bin/benchmark_model (Canonical tensorflow-lite-qcom-apps); the Google nightly linux_aarch64 binary will be downloaded instead"
command -v gst-ai-object-detection >/dev/null 2>&1 && info "IM SDK gst-ai-object-detection present (YOLOv8-style TFLite on runtime=dsp; not benchmarked here)" || true
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null); info "CPU governor: ${GOV:-?} (for peak: echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor)"

# ==================================================================
section "STEP 1: Python environment setup"
# ==================================================================
if [ -z "$PYBIN" ]; then for c in python3 python3.12 python3.11 python3.13; do command -v "$c" >/dev/null 2>&1 || continue; v=$("$c" -c 'import sys;print(sys.version_info.minor)'); [ "$v" -ge 11 ] && { PYBIN="$c"; break; }; done; fi
[ -n "$PYBIN" ] || { PYBIN=python3; warn "Python >= 3.11 not found — onnxruntime-qnn (NPU) unavailable"; }
info "Using $PYBIN ($($PYBIN --version 2>&1))"
if [ "$SKIP_INSTALL" -eq 0 ]; then
  [ -d "$VENV_DIR" ] || "$PYBIN" -m venv "$VENV_DIR" || { fail "venv failed (sudo apt install python3-venv)"; exit 1; }
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"; pip install -q --upgrade pip
  pip install -q ultralytics onnx onnxslim 2>"$LOGDIR/pip_ultralytics.log" && pass "ultralytics installed" || fail "ultralytics install failed — $LOGDIR/pip_ultralytics.log"
  pip install -q "onnxruntime>=1.24.1" "onnxruntime-qnn>=2.4.0" 2>"$LOGDIR/pip_ortqnn.log" && pass "onnxruntime + onnxruntime-qnn installed" || warn "onnxruntime-qnn install failed (needs Python >= 3.11, glibc >= 2.34) — $LOGDIR/pip_ortqnn.log"
  pip install -q ncnn 2>"$LOGDIR/pip_ncnn.log" && pass "ncnn installed" || warn "ncnn install failed"
  pip install -q ai-edge-litert 2>"$LOGDIR/pip_tflite.log" && pass "ai-edge-litert installed" || warn "ai-edge-litert install failed — TFLite rows skipped"
else
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"; info "Skipping installs"
fi
python3 - <<'PYEOF' 2>&1 | tee "$LOGDIR/versions.log"
import importlib
for m in ("torch", "ultralytics", "onnxruntime", "onnxruntime_qnn", "ncnn", "ai_edge_litert", "cv2", "numpy"):
    try: mod = importlib.import_module(m); print(f"  {m:<16} {getattr(mod, '__version__', 'ok')}")
    except Exception as e: print(f"  {m:<16} NOT AVAILABLE ({str(e)[:60]})")
PYEOF
TEST_IMG="$WORKDIR/bus.jpg"; [ -f "$TEST_IMG" ] || curl -sL -o "$TEST_IMG" https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg
[ -s "$TEST_IMG" ] && pass "Test image ready" || fail "Test image download failed"
TEST_VIDEO="$WORKDIR/solutions_ci_demo.mp4"; [ -f "$TEST_VIDEO" ] || curl -sL -o "$TEST_VIDEO" https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4
[ -s "$TEST_VIDEO" ] && { pass "Test video ready"; VIDEO_ARG="$TEST_VIDEO"; } || { fail "Test video download failed"; VIDEO_ARG=""; rm -f "$TEST_VIDEO"; }
declare -a BENCH_MODEL_NAMES=("yolo11n" "yolo11s" "yolo11m" "yolo26n" "yolo26s" "yolo26m"); [ "$QUICK_MODE" -eq 1 ] && BENCH_MODEL_NAMES=("yolo11n" "yolo26n")
for m in "${BENCH_MODEL_NAMES[@]}"; do [ -f "$WORKDIR/$m.pt" ] || ( cd "$WORKDIR" && python3 -c "from ultralytics import YOLO; YOLO('$m.pt')" >"$LOGDIR/dl_$m.log" 2>&1 ); [ -f "$WORKDIR/$m.pt" ] && pass "weights: $m.pt" || fail "download $m.pt failed"; done

# ==================================================================
section "STEP 2: Runtime visibility — ORT QNN EP (HTP/GPU/CPU), TFLite + QNN delegate, ncnn Vulkan"
# ==================================================================
QNN_HTP=0; QNN_GPU=0; QNN_CPU=0; ADSP_PATH=""
OUT=$(python3 - <<'PYEOF' 2>&1
import os, glob, time, numpy as np
try:
    import onnxruntime as ort, onnxruntime_qnn as q
except Exception as e:
    print(f"QNN_IMPORT_ERROR={e}"); raise SystemExit
print(f"ORT={ort.__version__} QNN_EP={getattr(q,'__version__','?')}")
pk = os.path.dirname(q.__file__)
skels = glob.glob(os.path.join(pk, "**", "libQnnHtpV68Skel.so"), recursive=True)
print(f"WHEEL_SKEL={skels[0] if skels else 'none'}")
adsp = os.environ.get("ADSP_LIBRARY_PATH") or (os.path.dirname(skels[0]) if skels else "")
if adsp: os.environ["ADSP_LIBRARY_PATH"] = adsp
print(f"ADSP_LIBRARY_PATH={adsp or '<none>'}")
ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())
try: print(f"EP_DEVICES={[d.ep_name for d in ort.get_ep_devices() if d.ep_name=='QNNExecutionProvider']}")
except Exception as e: print(f"EP_DEVICES_ERROR={e}")
# tiny conv graph
import onnx
from onnx import helper, TensorProto
w = np.random.rand(8, 3, 3, 3).astype(np.float32)
g = helper.make_graph([helper.make_node("Conv", ["x", "w"], ["y"], pads=[1,1,1,1])], "probe",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1,3,64,64])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1,8,64,64])],
    [helper.make_tensor("w", TensorProto.FLOAT, w.shape, w.flatten())])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 8
p = "/tmp/qnn_probe.onnx"; onnx.save(m, p)
for be in ("htp", "gpu", "cpu"):
    try:
        so = ort.SessionOptions(); so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        opts = {"backend_type": be, "htp_arch": "68"}
        if be == "htp": opts["enable_htp_fp16_precision"] = "1"
        t0 = time.perf_counter()
        s = ort.InferenceSession(p, so, providers=["QNNExecutionProvider"], provider_options=[opts])
        t1 = time.perf_counter(); y = s.run(None, {"x": np.random.rand(1,3,64,64).astype(np.float32)})[0]; t2 = time.perf_counter()
        print(f"QNN_PROBE_{be.upper()}=OK compile={1000*(t1-t0):.0f}ms infer={1000*(t2-t1):.2f}ms providers={s.get_providers()}")
    except Exception as e:
        print(f"QNN_PROBE_{be.upper()}=ERROR {str(e).splitlines()[0][:220]}")
PYEOF
)
echo "$OUT" | tee "$LOGDIR/qnn_probe.log"
if echo "$OUT" | grep -q "QNN_IMPORT_ERROR"; then warn "onnxruntime-qnn not importable — NPU via ORT unavailable"; else
  ADSP_PATH=$(echo "$OUT" | grep "^ADSP_LIBRARY_PATH=" | cut -d= -f2-); [ "$ADSP_PATH" != "<none>" ] && export ADSP_LIBRARY_PATH="$ADSP_PATH"
  for be in HTP GPU CPU; do
    l=$(echo "$OUT" | grep "^QNN_PROBE_${be}=")
    if echo "$l" | grep -q "=OK"; then pass "QNN $be backend works: $(echo "$l" | cut -d' ' -f2-3)"; case $be in HTP) QNN_HTP=1;; GPU) QNN_GPU=1;; CPU) QNN_CPU=1;; esac
    else case $be in HTP) fail "QNN HTP (NPU) probe failed: $(echo "$l" | cut -d' ' -f2- | cut -c1-200)";; GPU) warn "QNN GPU probe failed (expected on the Mesa image; needs proprietary Adreno OpenCL): $(echo "$l" | cut -d' ' -f2- | cut -c1-120)";; CPU) warn "QNN CPU backend probe failed: $(echo "$l" | cut -d' ' -f2- | cut -c1-120)";; esac; fi
  done
fi
TFLITE_OK=$(python3 -c "import ai_edge_litert.interpreter; print(1)" 2>/dev/null || echo 0); [ "$TFLITE_OK" = "1" ] && pass "TFLite runtime importable" || warn "no TFLite runtime"
QNN_DELEGATE=""; for d in "$QNN_SYS_LIB" /usr/lib /usr/lib/aarch64-linux-gnu; do [ -n "$d" ] && [ -f "$d/libQnnTFLiteDelegate.so" ] && { QNN_DELEGATE="$d/libQnnTFLiteDelegate.so"; break; }; done
[ -n "$QNN_DELEGATE" ] && pass "QNN TFLite delegate: $QNN_DELEGATE" || info "libQnnTFLiteDelegate.so not found — install qairt-libs (Canonical PPA) or set QNN_SDK_ROOT; TFLite-on-NPU rows skipped"
NCNN_GPU=0; OUT=$(python3 -c "import ncnn; n=ncnn.get_gpu_count(); print('NCNN_GPU_COUNT=%d'%n); print('NCNN_GPU_NAME='+ncnn.get_gpu_info(0).device_name()) if n else None" 2>&1); echo "$OUT" > "$LOGDIR/ncnn_probe.log"
echo "$OUT" | grep -q "NCNN_GPU_COUNT=[1-9]" && { pass "ncnn sees Vulkan GPU: $(echo "$OUT" | grep NCNN_GPU_NAME | cut -d= -f2-)"; NCNN_GPU=1; } || warn "ncnn has no Vulkan device (install mesa-vulkan-drivers; turnip supports Adreno 643)"
OPENCV_OCL=$(python3 -c "import cv2; print(int(cv2.ocl.haveOpenCL()))" 2>/dev/null || echo 0); [ "$OPENCV_OCL" = "1" ] && pass "OpenCV sees OpenCL" || info "OpenCV: no OpenCL"

# ==================================================================
section "STEP 3: Models — tarball or on-board export"
# ==================================================================
if [ -n "$MODELS_IN" ]; then
  if [ -f "$MODELS_IN" ]; then tar -C "$WORKDIR" -xzf "$MODELS_IN" && pass "Unpacked $MODELS_IN" || fail "Could not unpack $MODELS_IN"
  elif [ -d "$MODELS_IN" ]; then cp -r "$MODELS_IN"/* "$WORKDIR"/ && pass "Copied models from $MODELS_IN"; else fail "--models not found: $MODELS_IN"; fi
fi
( cd "$WORKDIR" && for m in "${BENCH_MODEL_NAMES[@]}"; do
  [ -f "$m.pt" ] || continue
  [ -f "$m.onnx" ] || python3 -c "from ultralytics import YOLO; YOLO('$m.pt').export(format='onnx', imgsz=640, dynamic=False, simplify=True)" >"$LOGDIR/export_onnx_$m.log" 2>&1
  [ -d "${m}_ncnn_model" ] || python3 -c "from ultralytics import YOLO; YOLO('$m.pt').export(format='ncnn', imgsz=640)" >"$LOGDIR/export_ncnn_$m.log" 2>&1
  if [ ! -f "${m}_qnn.onnx" ] && python3 -c "import onnxruntime_qnn" 2>/dev/null; then
    echo "[INFO] Exporting $m -> QNN HTP v68 context (w8a16, coco8 calibration; can take several minutes on the board)..."
    python3 -c "from ultralytics import YOLO; YOLO('$m.pt').export(format='qnn', name='68', imgsz=640)" >"$LOGDIR/export_qnn_$m.log" 2>&1
  fi
done )
for m in "${BENCH_MODEL_NAMES[@]}"; do
  [ -f "$WORKDIR/$m.onnx" ] && pass "ONNX: $m.onnx" || warn "ONNX missing for $m"
  [ -d "$WORKDIR/${m}_ncnn_model" ] && pass "NCNN: ${m}_ncnn_model" || warn "NCNN missing for $m"
  [ -f "$WORKDIR/${m}_qnn.onnx" ] && pass "QNN HTP v68 context: ${m}_qnn.onnx" || warn "QNN export missing for $m (see $LOGDIR/export_qnn_$m.log; needs onnxruntime-qnn and Python >= 3.11)"
  ls "$WORKDIR/${m}_saved_model"/*.tflite >/dev/null 2>&1 && pass "TFLite: $(ls "$WORKDIR/${m}_saved_model"/*.tflite | xargs -n1 basename | tr '\n' ' ')" || info "no TFLite for $m (export on a PC: ./export_host.sh)"
done

# ==================================================================
section "STEP 4: Full benchmark — every model x every working backend (end-to-end, with NMS)"
# ==================================================================
run_benchmark() {
  local backend_label="$1" device_arg="$2" model_name="$3" model_path="$4" severity="${5:-fail}"
  [ -e "$model_path" ] || { warn "Model $model_path missing, skipping ${backend_label}/${model_name}"; return; }
  info "Benchmarking [$backend_label] model=$model_name ..."
  local logfile="$LOGDIR/bench_${backend_label}_${model_name}.log"; local OUT
  OUT=$(python3 "$SCRIPT_DIR/bench_harness.py" --model "$model_path" ${device_arg:+--device "$device_arg"} --image "$TEST_IMG" ${VIDEO_ARG:+--video "$VIDEO_ARG"} --backend-label "$backend_label" --img-runs 15 --video-max-frames 300 2>&1)
  echo "$OUT" > "$logfile"
  if echo "$OUT" | grep -q "^STEADY_AVG_MS="; then
    local cold first steady_avg steady_p95 vfps vms
    cold=$(echo "$OUT" | grep "^COLD_LOAD_MS=" | cut -d= -f2); first=$(echo "$OUT" | grep "^FIRST_INFER_MS=" | cut -d= -f2)
    steady_avg=$(echo "$OUT" | grep "^STEADY_AVG_MS=" | cut -d= -f2); steady_p95=$(echo "$OUT" | grep "^STEADY_P95_MS=" | cut -d= -f2)
    vfps=$(echo "$OUT" | grep "^VIDEO_AVG_FPS=" | cut -d= -f2); vms=$(echo "$OUT" | grep "^VIDEO_AVG_MS=" | cut -d= -f2)
    pass "[$backend_label/$model_name] cold-load ${cold}ms | first-infer ${first}ms | steady ${steady_avg}ms (p95 ${steady_p95}ms) | video ${vfps:-N/A} FPS (${vms:-N/A}ms/frame) | dets $(echo "$OUT" | grep '^STEADY_DETECTIONS=' | cut -d= -f2)"
    { echo "$backend_label,$model_name,cold_load,ms,$cold"; echo "$backend_label,$model_name,first_infer,ms,$first"; echo "$backend_label,$model_name,steady_avg,ms,$steady_avg"; echo "$backend_label,$model_name,steady_p95,ms,$steady_p95"
      [ -n "$vfps" ] && echo "$backend_label,$model_name,video,fps,$vfps"; [ -n "$vms" ] && echo "$backend_label,$model_name,video,ms_per_frame,$vms"; } >> "$BENCH_CSV"
  else
    local err; err=$(echo "$OUT" | grep -E "_ERROR=" | head -1 | cut -c1-180)
    [ "$severity" = "warn" ] && warn "[$backend_label/$model_name] did not run (${err:-see $logfile})" || fail "[$backend_label/$model_name] failed — ${err:-see $logfile}"
  fi
}
for m in "${BENCH_MODEL_NAMES[@]}"; do run_benchmark "pytorch_cpu" "cpu" "$m" "$WORKDIR/$m.pt"; done
if [ "$QNN_HTP" -eq 1 ]; then for m in "${BENCH_MODEL_NAMES[@]}"; do run_benchmark "qnn_npu_w8a16" "" "$m" "$WORKDIR/${m}_qnn.onnx"; done; else warn "Skipping Ultralytics QNN (NPU) benchmarks — HTP probe failed in Step 2"; fi
for m in "${BENCH_MODEL_NAMES[@]}"; do run_benchmark "onnxrt_cpu" "cpu" "$m" "$WORKDIR/$m.onnx" warn; done
for m in "${BENCH_MODEL_NAMES[@]}"; do run_benchmark "ncnn_cpu" "cpu" "$m" "$WORKDIR/${m}_ncnn_model"; done
if [ "$NCNN_GPU" -eq 1 ]; then for m in "${BENCH_MODEL_NAMES[@]}"; do run_benchmark "ncnn_vulkan" "vulkan:0" "$m" "$WORKDIR/${m}_ncnn_model" warn; done; else warn "Skipping ncnn Vulkan (Adreno) — no Vulkan device"; fi
if [ "$TFLITE_OK" = "1" ]; then for m in "${BENCH_MODEL_NAMES[@]}"; do
  f=$(ls "$WORKDIR/${m}_saved_model"/*_float32.tflite 2>/dev/null | head -1); [ -n "$f" ] && run_benchmark "tflite_cpu_fp32" "cpu" "$m" "$f" warn
  f=$(ls "$WORKDIR/${m}_saved_model"/*_full_integer_quant.tflite 2>/dev/null | head -1); [ -n "$f" ] && run_benchmark "tflite_cpu_int8" "cpu" "$m" "$f" warn
done; fi

# ==================================================================
section "STEP 5: Raw ONNX Runtime + QNN EP — HTP / GPU / QNN-CPU / ORT-CPU (no NMS)"
# ==================================================================
run_ort() { # label model backend extra
  local label="$1" f="$2" be="$3" extra="$4" m; m=$(basename "$f" .onnx | sed 's/_qnn$//')
  local OUT; OUT=$(python3 "$SCRIPT_DIR/ort_qnn_bench.py" --model "$f" --backend "$be" $extra ${VIDEO_ARG:+--video "$VIDEO_ARG"} --runs 15 --video-max-frames 300 2>&1)
  echo "$OUT" > "$LOGDIR/${label}_${m}.log"
  if echo "$OUT" | grep -q "^STEADY_AVG_MS="; then
    local sess avg vfps; sess=$(echo "$OUT" | grep "^SESSION_CREATE_MS=" | cut -d= -f2); avg=$(echo "$OUT" | grep "^STEADY_AVG_MS=" | cut -d= -f2); vfps=$(echo "$OUT" | grep "^VIDEO_AVG_FPS=" | cut -d= -f2)
    pass "[$label/$m] session ${sess}ms | raw infer ${avg}ms | video ${vfps:-N/A} FPS | $(echo "$OUT" | grep '^ACTUAL_PROVIDERS=' | cut -d= -f2-)"
    { echo "$label,$m,cold_load,ms,$sess"; echo "$label,$m,steady_avg,ms,$avg"; [ -n "$vfps" ] && echo "$label,$m,video,fps,$vfps"; } >> "$BENCH_CSV"
  else warn "[$label/$m] $(echo "$OUT" | grep -E '_ERROR=|^ERROR=' | head -1 | cut -c1-200)"; fi
}
for m in yolo11n yolo26n; do
  [ -f "$WORKDIR/$m.onnx" ] && run_ort "ort_cpu_raw" "$WORKDIR/$m.onnx" ort-cpu ""
  if python3 -c "import onnxruntime_qnn" 2>/dev/null; then
    [ "$QNN_HTP" -eq 1 ] && [ -f "$WORKDIR/$m.onnx" ] && run_ort "ort_qnn_htp_fp16_plainonnx" "$WORKDIR/$m.onnx" htp "--fp16 --allow-fallback"
    [ "$QNN_GPU" -eq 1 ] && [ -f "$WORKDIR/$m.onnx" ] && run_ort "ort_qnn_gpu_plainonnx" "$WORKDIR/$m.onnx" gpu "--allow-fallback"
    [ "$QNN_CPU" -eq 1 ] && [ -f "$WORKDIR/$m.onnx" ] && run_ort "ort_qnn_cpubackend" "$WORKDIR/$m.onnx" cpu ""
    if [ "$QNN_HTP" -eq 1 ] && [ -f "$WORKDIR/${m}_qnn.onnx" ]; then
      for pm in burst balanced default; do run_ort "ort_qnn_htp_ctx_${pm}" "$WORKDIR/${m}_qnn.onnx" htp "--perf-mode $pm"; done
    fi
  fi
done

# ==================================================================
section "STEP 6: Raw TFLite — XNNPACK / QNN delegate (HTP, GPU) / benchmark_model"
# ==================================================================
run_tfl() { # label file delegate opts...
  local label="$1" f="$2" del="$3"; shift 3; local m; m=$(basename "$f" .tflite | sed 's/_.*//')
  local OUT; OUT=$(python3 "$SCRIPT_DIR/tflite_bench.py" --model "$f" --threads 8 ${del:+--delegate "$del"} "$@" ${VIDEO_ARG:+--video "$VIDEO_ARG"} --runs 15 --video-max-frames 300 2>&1)
  echo "$OUT" > "$LOGDIR/${label}_$(basename "$f" .tflite).log"
  if echo "$OUT" | grep -q "^STEADY_AVG_MS="; then
    local avg init vfps; avg=$(echo "$OUT" | grep "^STEADY_AVG_MS=" | cut -d= -f2); init=$(echo "$OUT" | grep "^INTERP_INIT_MS=" | cut -d= -f2); vfps=$(echo "$OUT" | grep "^VIDEO_AVG_FPS=" | cut -d= -f2)
    pass "[$label/$(basename "$f")] init ${init}ms | raw infer ${avg}ms | video ${vfps:-N/A} FPS"
    { echo "$label,$m,first_infer,ms,$init"; echo "$label,$m,steady_avg,ms,$avg"; [ -n "$vfps" ] && echo "$label,$m,video,fps,$vfps"; } >> "$BENCH_CSV"
  else warn "[$label/$(basename "$f")] $(echo "$OUT" | grep -E '_ERROR=|^ERROR=' | head -1 | cut -c1-200)"; fi
}
BM=$(command -v benchmark_model || true)
if [ -z "$BM" ]; then BM="$WORKDIR/linux_aarch64_benchmark_model"; [ -s "$BM" ] || curl -fsSL -o "$BM" https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model && chmod +x "$BM"; [ -x "$BM" ] || BM=""; fi
run_bm() { # label file flags
  local label="$1" f="$2" flags="$3" m; m=$(basename "$f" .tflite | sed 's/_.*//')
  local OUT; OUT=$("$BM" --graph="$f" --num_runs=50 --warmup_runs=5 $flags 2>&1); echo "$OUT" > "$LOGDIR/${label}_$(basename "$f" .tflite).log"
  local avg; avg=$(echo "$OUT" | grep -oE "Inference \(avg\): [0-9.]+" | grep -oE "[0-9.]+$")
  if [ -n "$avg" ]; then local ms fps; ms=$(awk "BEGIN{printf \"%.2f\", $avg/1000}"); fps=$(awk "BEGIN{printf \"%.2f\", 1000000/$avg}")
    pass "[$label/$(basename "$f")] ${ms}ms avg (${fps} FPS) $(echo "$OUT" | grep -oE '(completely|partially) executed by the delegate[^.]*|not be executed by the delegate' | head -1 | sed 's/^/| /')"
    { echo "$label,$m,steady_avg,ms,$ms"; echo "$label,$m,video,fps,$fps"; } >> "$BENCH_CSV"
  else warn "[$label/$(basename "$f")] failed: $(echo "$OUT" | grep -iE 'error|fail' | head -1 | cut -c1-160)"; fi
}
if [ "$TFLITE_OK" = "1" ]; then for m in yolo11n yolo26n; do
  for f in "$WORKDIR/${m}_saved_model"/*_float32.tflite "$WORKDIR/${m}_saved_model"/*_full_integer_quant.tflite; do
    [ -f "$f" ] || continue; isint=0; [[ "$f" == *integer_quant* ]] && isint=1
    run_tfl "tflite_raw_cpu" "$f" ""
    if [ -n "$QNN_DELEGATE" ] && [ "$NPU_PLUMBING" -eq 1 ]; then
      run_tfl "tflite_raw_qnn_htp" "$f" "$QNN_DELEGATE" --delegate-opt backend_type=htp --delegate-opt "library_path=$QNN_SYS_LIB/libQnnHtp.so" --delegate-opt "skel_library_dir=${SKEL_SYS:-$ADSP_PATH}" --delegate-opt htp_performance_mode=2 --delegate-opt "htp_precision=$([ $isint -eq 1 ] && echo 0 || echo 1)"
      [ "$OPENCL_KIND" = "adreno" ] && run_tfl "tflite_raw_qnn_gpu" "$f" "$QNN_DELEGATE" --delegate-opt backend_type=gpu --delegate-opt "library_path=$QNN_SYS_LIB/libQnnGpu.so"
    fi
    if [ -n "$BM" ]; then
      run_bm "benchmark_model_cpu" "$f" "--use_xnnpack=true --num_threads=8"
      [ "$OPENCL_KIND" != "none" ] && run_bm "benchmark_model_gpu" "$f" "--use_gpu=true --gpu_precision_loss_allowed=true"
      [ -n "$QNN_DELEGATE" ] && [ "$NPU_PLUMBING" -eq 1 ] && run_bm "benchmark_model_qnn_htp" "$f" "--external_delegate_path=$QNN_DELEGATE --external_delegate_options=backend_type:htp;library_path:$QNN_SYS_LIB/libQnnHtp.so;skel_library_dir:${SKEL_SYS:-$ADSP_PATH};htp_precision:$([ $isint -eq 1 ] && echo 0 || echo 1);htp_performance_mode:2"
    fi
  done
done; fi

# ==================================================================
section "STEP 7: NPU diagnosis"
# ==================================================================
if grep -q "^\[PASS\] \[qnn_npu_" "$RESULTS_FILE"; then pass "Hexagon HTP NPU usable end-to-end via Ultralytics QNN export + onnxruntime-qnn"
elif [ ! -e /dev/fastrpc-cdsp ]; then warn "ROOT CAUSE: no /dev/fastrpc-cdsp. Kernel/DT problem: use Radxa's R2+ image kernel (6.18-qcom with fastrpc memory-region) or Canonical's linux-image-qcom; check 'dmesg | grep -i fastrpc' for 'no reserved DMA memory'"
elif ! grep -q "running" <(cat /sys/class/remoteproc/remoteproc*/state 2>/dev/null); then warn "ROOT CAUSE: cdsp/adsp remoteproc not running — DSP firmware missing/corrupt: sudo apt install --reinstall radxa-firmware-qcs6490 (or linux-firmware-dragonwing), then reboot"
elif [ -z "$LIBCDSP" ]; then warn "ROOT CAUSE: libcdsprpc.so missing — sudo apt install fastrpc libcdsprpc1 (Radxa) / qcom-fastrpc1 (Canonical PPA)"
elif ! { [ -r /dev/fastrpc-cdsp ] && [ -w /dev/fastrpc-cdsp ]; }; then warn "ROOT CAUSE: /dev/fastrpc-cdsp permissions — echo 'KERNEL==\"fastrpc-*\", MODE=\"0666\"' | sudo tee /etc/udev/rules.d/99-fastrpc.rules; echo 'SUBSYSTEM==\"dma_heap\", KERNEL==\"system\", MODE=\"0666\"' | sudo tee -a /etc/udev/rules.d/99-fastrpc.rules; sudo udevadm trigger"
elif grep -qiE "0x80000600|14001|Failed to load skel" "$LOGDIR/qnn_probe.log"; then warn "ROOT CAUSE: FastRPC session/skel failure (0x80000600 / 14001) — usually a skel vs runtime version mismatch or DSP-side libc++ missing. Ensure ADSP_LIBRARY_PATH points at the skel shipped with the SAME QAIRT version as the runtime (the onnxruntime-qnn wheel bundles a matching pair: $ADSP_PATH); for system QAIRT, match qairt-dsp-binaries to qairt-libs"
elif ! python3 -c "import onnxruntime_qnn" 2>/dev/null; then warn "ROOT CAUSE: onnxruntime-qnn not installed (Python >= 3.11, glibc >= 2.34 aarch64) — see $LOGDIR/pip_ortqnn.log"
else warn "NPU path failed for another reason — see $LOGDIR/qnn_probe.log, $LOGDIR/dmesg_npu.log and bench_qnn_npu_*.log"; fi
[ "$QNN_GPU" -eq 1 ] || info "QNN GPU backend / TFLite GPU delegate need the proprietary Adreno OpenCL (qcom-adreno-cl1 from ppa:ubuntu-qcom-iot/qcom-ppa); on the Mesa image use NCNN Vulkan (turnip) for GPU inference"

# ==================================================================
section "SUMMARY"
# ==================================================================
echo -e "\nResults: $RESULTS_FILE | logs: $LOGDIR | CSV: $BENCH_CSV"
echo -e "${GREEN}Passed: $(grep -c '^\[PASS\]' "$RESULTS_FILE")${NC}  ${YELLOW}Warnings: $(grep -c '^\[WARN\]' "$RESULTS_FILE")${NC}  ${RED}Failed: $(grep -c '^\[FAIL\]' "$RESULTS_FILE")${NC}\n"; cat "$RESULTS_FILE"
table() { echo -e "\n${BLUE}--------------------------------------------------------------${NC}\n${BLUE}  $2${NC}\n${BLUE}--------------------------------------------------------------${NC}"; printf "%-34s %-10s %12s\n" "BACKEND" "MODEL" "$3"; grep ",$1," "$BENCH_CSV" | sort -t, -k2,2 -k1,1 | while IFS=, read -r b m s met v; do printf "%-34s %-10s %12s\n" "$b" "$m" "$v"; done; }
table "video,fps" "VIDEO FPS (end-to-end: pytorch_/qnn_npu_/onnxrt_cpu/ncnn_/tflite_cpu_; raw: ort_*/tflite_raw_*/benchmark_model_*) — higher is better" "VIDEO_FPS"
table "steady_avg,ms" "STEADY-STATE LATENCY (ms/frame) — lower is better" "MS/FRAME"
table "first_infer,ms" "FIRST INFERENCE / INIT (ms)" "MS"
table "cold_load,ms" "COLD LOAD / SESSION CREATE (ms)" "MS"
ENV_FILE="$WORKDIR/yolo_q6a_env.sh"; { echo "# Auto-generated by test_yolo_q6a.sh on $(date) — soc_id ${SOC_ID:-?}, kernel $(uname -r)"; [ -n "$ADSP_PATH" ] && [ "$ADSP_PATH" != "<none>" ] && echo "export ADSP_LIBRARY_PATH=$ADSP_PATH"; } > "$ENV_FILE"
BEST=$(grep -E "^(pytorch_cpu|qnn_npu_w8a16|onnxrt_cpu|ncnn_cpu|ncnn_vulkan|tflite_cpu_(fp32|int8)),yolo(11|26)n,video,fps," "$BENCH_CSV" | sort -t, -k5,5 -gr | head -1)
if [ -n "$BEST" ]; then bb=$(echo "$BEST" | cut -d, -f1); bm=$(echo "$BEST" | cut -d, -f2); bf=$(echo "$BEST" | cut -d, -f5)
  echo -e "\n${GREEN}RECOMMENDATION:${NC} fastest end-to-end nano backend: $bb ($bm, $bf FPS)"
  case "$bb" in qnn_npu_*) echo "export YOLO_MODEL=$WORKDIR/${bm}_qnn.onnx   # yolo predict model=\$YOLO_MODEL source=img.jpg   (export: yolo export model=${bm}.pt format=qnn name=68)" ;;
    ncnn_vulkan) echo "export YOLO_MODEL=$WORKDIR/${bm}_ncnn_model; export YOLO_DEVICE=vulkan:0" ;; ncnn_cpu) echo "export YOLO_MODEL=$WORKDIR/${bm}_ncnn_model; export YOLO_DEVICE=cpu" ;;
    onnxrt_cpu) echo "export YOLO_MODEL=$WORKDIR/${bm}.onnx" ;; tflite_cpu_*) echo "export YOLO_MODEL=\$(ls $WORKDIR/${bm}_saved_model/*.tflite | head -1)" ;; *) echo "export YOLO_MODEL=$WORKDIR/${bm}.pt; export YOLO_DEVICE=cpu" ;; esac >> "$ENV_FILE"
else echo -e "\n${YELLOW}RECOMMENDATION:${NC} no benchmark completed — fix the FAIL items first."; fi
echo -e "${BLUE}Env file: $ENV_FILE${NC}"; echo "---"; cat "$ENV_FILE"; echo "---"
