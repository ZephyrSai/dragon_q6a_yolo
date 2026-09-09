#!/usr/bin/env bash
#
# export_host.sh — export YOLO11 / YOLO26 on a PC for the Dragon Q6A suite
# and pack them into one tarball.
#
# Why a PC step at all: Ultralytics' LiteRT/TFLite export needs the
# litert-converter (x86-64 / macOS only), so the .tflite files used by the
# QNN TFLite delegate, benchmark_model and Android must be produced here.
# QNN (HTP v68 context binary), ONNX and NCNN exports also work on the board
# itself, but doing them here is faster; the board script fills in anything
# missing.
#
# Produces per model (yolo11n/s/m, yolo26n/s/m; --quick = nano only):
#   <m>.onnx                       plain FP32 ONNX (ORT CPU, QNN EP compile, OpenCV DNN)
#   <m>_qnn.onnx                   Ultralytics QNN export: precompiled HTP v68 context (w8a16)
#   <m>_ncnn_model/                NCNN (CPU + Vulkan/turnip on Adreno)
#   <m>_saved_model/<m>_float32.tflite, <m>_full_integer_quant.tflite   (coco8 calibration)
#
# Usage:
#   ./export_host.sh [--quick] [--no-qnn] [--no-tflite] [--workdir DIR] [--python python3.12]
#   -> <workdir>/q6a_models.tar.gz

set -uo pipefail
WORKDIR="$HOME/yolo-q6a-export"; QUICK=0; DO_QNN=1; DO_TFLITE=1; PYBIN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=1 ;; --no-qnn) DO_QNN=0 ;; --no-tflite) DO_TFLITE=0 ;;
    --workdir) WORKDIR="$2"; shift ;; --python) PYBIN="$2"; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
  esac; shift
done
mkdir -p "$WORKDIR/logs"; LOGDIR="$WORKDIR/logs"
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
pass() { echo -e "${GREEN}[PASS]${NC} $1"; }; fail() { echo -e "${RED}[FAIL]${NC} $1"; }; warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }; info() { echo -e "${BLUE}[INFO]${NC} $1"; }

ARCH=$(uname -m); OS=$(uname -s)
[ "$ARCH" = "aarch64" ] && [ "$DO_TFLITE" -eq 1 ] && { warn "TFLite export needs x86-64/macOS (litert-converter); disabling it here"; DO_TFLITE=0; }
[ "$OS" = "Darwin" ] && [ "$DO_QNN" -eq 1 ] && { warn "QNN export is not supported on macOS; disabling (run on the board or a Linux PC)"; DO_QNN=0; }

# onnxruntime-qnn (needed by the QNN export) wants Python >= 3.11
if [ -z "$PYBIN" ]; then
  for c in python3.12 python3.11 python3.13 python3; do
    command -v "$c" >/dev/null 2>&1 || continue
    v=$("$c" -c 'import sys;print(sys.version_info.minor)'); [ "$v" -ge 11 ] && { PYBIN="$c"; break; }
  done
fi
[ -n "$PYBIN" ] || { fail "Need Python >= 3.11 (for onnxruntime-qnn). Pass --python"; exit 1; }
info "arch=$ARCH os=$OS python=$PYBIN qnn=$DO_QNN tflite=$DO_TFLITE workdir=$WORKDIR"

VENV="$WORKDIR/venv-export"
[ -d "$VENV" ] || "$PYBIN" -m venv "$VENV" || { fail "venv failed"; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"; pip install -q --upgrade pip
pip install -q ultralytics onnx onnxslim 2>"$LOGDIR/pip.log" && pass "ultralytics ready" || { fail "pip failed: $LOGDIR/pip.log"; exit 1; }
[ "$DO_QNN" -eq 1 ] && { pip install -q "onnxruntime>=1.24.1" "onnxruntime-qnn>=2.4.0" 2>>"$LOGDIR/pip.log" && pass "onnxruntime-qnn ready" || { warn "onnxruntime-qnn install failed (see $LOGDIR/pip.log) — QNN export disabled"; DO_QNN=0; }; }
[ "$DO_TFLITE" -eq 1 ] && { pip install -q "tensorflow>=2.13" onnx2tf sng4onnx onnx_graphsurgeon ai-edge-litert 2>>"$LOGDIR/pip.log" || warn "TFLite deps failed to preinstall; Ultralytics will try again during export"; }

MODELS=("yolo11n" "yolo11s" "yolo11m" "yolo26n" "yolo26s" "yolo26m"); [ "$QUICK" -eq 1 ] && MODELS=("yolo11n" "yolo26n")
cd "$WORKDIR" || exit 1
cat > _export.py <<'PYEOF'
import sys
from ultralytics import YOLO
m, fmt, prec = sys.argv[1:4]
model = YOLO(f"{m}.pt")
kw = dict(format=fmt, imgsz=640, batch=1)
if fmt == "qnn": kw["name"] = "68"                      # Hexagon v68 = QCS6490
if fmt == "onnx": kw.update(dynamic=False, simplify=True)
tries = {"fp32": [dict()], "int8": [dict(quantize=8, data="coco8.yaml"), dict(int8=True, data="coco8.yaml")]}[prec]
for extra in tries:
    try:
        out = model.export(**kw, **extra); print(f"EXPORTED={out}"); break
    except (TypeError, SyntaxError, KeyError) as e:
        print(f"args {extra} rejected: {e}")
else:
    raise SystemExit("EXPORT_ERROR")
PYEOF
ex() { # model fmt prec check_path label
  [ -e "$4" ] && { info "exists: $4"; return; }
  info "Exporting $5 ..."; python3 _export.py "$1" "$2" "$3" >"$LOGDIR/export_${5//\//_}.log" 2>&1
  [ -e "$4" ] && pass "$5 -> $4" || fail "$5 failed — $LOGDIR/export_${5//\//_}.log: $(grep -m1 -iE 'error' "$LOGDIR/export_${5//\//_}.log" | cut -c1-140)"
}
for m in "${MODELS[@]}"; do
  [ -f "$m.pt" ] || python3 -c "from ultralytics import YOLO; YOLO('$m.pt')" >"$LOGDIR/dl_$m.log" 2>&1
  [ -f "$m.pt" ] || { fail "download $m.pt"; continue; }
  ex "$m" onnx fp32 "$m.onnx" "$m/onnx"
  ex "$m" ncnn fp32 "${m}_ncnn_model" "$m/ncnn"
  [ "$DO_QNN" -eq 1 ] && ex "$m" qnn fp32 "${m}_qnn.onnx" "$m/qnn-htp-v68"
  if [ "$DO_TFLITE" -eq 1 ]; then
    ex "$m" tflite fp32 "${m}_saved_model/${m}_float32.tflite" "$m/tflite/fp32"
    ex "$m" tflite int8 "${m}_saved_model/${m}_full_integer_quant.tflite" "$m/tflite/int8"
  fi
done
items=(); for m in "${MODELS[@]}"; do for d in "$m.onnx" "${m}_qnn.onnx" "${m}_ncnn_model" "${m}_saved_model"; do [ -e "$d" ] && items+=("$d"); done; done
[ ${#items[@]} -gt 0 ] && tar -czf q6a_models.tar.gz "${items[@]}" && pass "Packed ${#items[@]} items -> $WORKDIR/q6a_models.tar.gz ($(du -h q6a_models.tar.gz | cut -f1))"
info "Board: ./test_yolo_q6a.sh --models q6a_models.tar.gz   |  Android: ./android/test_yolo_q6a_android.sh --models-dir $WORKDIR"
