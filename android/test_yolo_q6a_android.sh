#!/usr/bin/env bash
#
# android/test_yolo_q6a_android.sh — run ON A HOST PC with adb, against a
# Radxa Dragon Q6A (QCS6490) running Android (Radxa's Android 15 image).
#
# What it does:
#   0. Device identity (all the SoC-revealing props), Qualcomm AI stack in
#      /vendor (libQnnHtp*.so, libcdsprpc.so, /vendor/dsp/cdsp skels,
#      /dev/fastrpc-*), NNAPI vendor HALs (qti-dsp / qti-gpu / qti-default)
#   1. Google's prebuilt TFLite benchmark_model (android_aarch64) on every
#      exported .tflite: CPU (XNNPACK), GPU delegate (OpenCL on Adreno),
#      NNAPI with qti-dsp / qti-gpu / default accelerators, and — when QNN
#      libs are supplied — the QNN TFLite delegate on HTP (NPU) and GPU.
#   2. qnn-net-run on an Ultralytics QNN context (optional, needs the QAIRT
#      SDK's Android bin + libs and a context binary extracted from the
#      *_qnn.onnx wrapper) — reported as informational.
#   3. Summary table + CSV (same schema as the Linux suite).
#
# Usage:
#   ./android/test_yolo_q6a_android.sh --models-dir ~/yolo-q6a-export [--serial S]
#        [--qnn-android-libs <QAIRT>/lib/aarch64-android] [--qnn-skel <QAIRT>/lib/hexagon-v68/unsigned]
#
# --models-dir: outputs of export_host.sh (*_saved_model/*.tflite).
# --qnn-android-libs / --qnn-skel: from the Qualcomm AI Runtime (QAIRT) SDK
#   zip (needs a Qualcomm ID to download). Without them only CPU/GPU/NNAPI run.

set -uo pipefail
MODELS_DIR="$HOME/yolo-q6a-export"; SERIAL=""; QNN_LIBS=""; QNN_SKEL=""
WORKDIR="$HOME/yolo-q6a-android-test"; DEV_DIR="/data/local/tmp/yolo_bench"
BM_URL="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model"
while [ $# -gt 0 ]; do case "$1" in
  --models-dir) MODELS_DIR="$2"; shift ;; --serial) SERIAL="$2"; shift ;;
  --qnn-android-libs) QNN_LIBS="$2"; shift ;; --qnn-skel) QNN_SKEL="$2"; shift ;;
  -h|--help) sed -n '2,28p' "$0"; exit 0 ;; esac; shift; done
mkdir -p "$WORKDIR/logs"; LOGDIR="$WORKDIR/logs"; RESULTS_FILE="$WORKDIR/results_summary.txt"; BENCH_CSV="$WORKDIR/benchmark_results.csv"
: > "$RESULTS_FILE"; echo "backend,model,stage,metric,value_ms_or_fps" > "$BENCH_CSV"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
section() { echo -e "\n${BLUE}==================================================================${NC}\n${BLUE}  $1${NC}\n${BLUE}==================================================================${NC}"; }
pass() { echo -e "${GREEN}[PASS]${NC} $1"; echo "[PASS] $1" >> "$RESULTS_FILE"; }; fail() { echo -e "${RED}[FAIL]${NC} $1"; echo "[FAIL] $1" >> "$RESULTS_FILE"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; echo "[WARN] $1" >> "$RESULTS_FILE"; }; info() { echo -e "${BLUE}[INFO]${NC} $1"; }
ADB="adb${SERIAL:+ -s $SERIAL}"; ash() { $ADB shell "$@" | tr -d '\r'; }

section "STEP 0: Device identity + Qualcomm AI stack"
command -v adb >/dev/null || { fail "adb not found"; exit 1; }
$ADB get-state >/dev/null 2>&1 || { fail "No adb device (Radxa's AOSP image only exposes adb over TCP: adb connect <ip>:5555)"; exit 1; }
for p in ro.board.platform ro.hardware ro.product.board ro.soc.model ro.soc.manufacturer ro.product.model ro.build.version.release ro.product.cpu.abi; do echo "$p=$(ash getprop $p)"; done | tee "$LOGDIR/device.log"
SOCID=$(ash "cat /sys/devices/soc0/soc_id 2>/dev/null"); COMPAT=$(ash "cat /proc/device-tree/compatible 2>/dev/null | tr '\0' ' '")
echo "soc_id=$SOCID compat=$COMPAT" | tee -a "$LOGDIR/device.log"
if [ "$SOCID" = "498" ] || [ "$SOCID" = "497" ] || echo "$(cat "$LOGDIR/device.log")" | grep -qiE "lahaina|kodiak|yupik|qcs6490|qcm6490|sm7325|6490"; then pass "QCS6490-family Android device ($(ash getprop ro.product.model), Android $(ash getprop ro.build.version.release))"; else warn "Could not confirm QCS6490 from props/soc_id — continuing"; fi
VQNN=$(ash "ls /vendor/lib64/libQnnHtp.so /vendor/lib64/libQnnHtpV68Stub.so /vendor/lib64/libcdsprpc.so 2>/dev/null" | tr '\n' ' '); [ -n "$VQNN" ] && pass "vendor QNN/FastRPC libs: $VQNN" || info "no /vendor/lib64/libQnnHtp*.so — QNN must be supplied via --qnn-android-libs"
VSKEL=$(ash "ls /vendor/dsp/cdsp/libQnnHtpV68Skel.so /vendor/lib/rfsa/adsp/libQnnHtpV68Skel.so 2>/dev/null" | tr '\n' ' '); [ -n "$VSKEL" ] && info "vendor HTP skel: $VSKEL"
ash "ls /dev/fastrpc-cdsp 2>/dev/null" | grep -q fastrpc && pass "/dev/fastrpc-cdsp present" || warn "no /dev/fastrpc-cdsp (AOSP/GloDroid builds have no NPU stack)"
NNHAL=$(ash "ls /vendor/bin/hw 2>/dev/null | grep -i neuralnetworks"); [ -n "$NNHAL" ] && pass "NNAPI vendor HAL(s): $(echo "$NNHAL" | tr '\n' ' ')" || warn "No NNAPI vendor HAL — --use_nnapi will run on nnapi-reference (CPU)"
ash "ls /vendor/lib64/libOpenCL.so 2>/dev/null" | grep -q OpenCL && pass "Adreno OpenCL present (/vendor/lib64/libOpenCL.so)" || warn "no /vendor/lib64/libOpenCL.so — GPU delegate falls back to GLES or fails"

section "STEP 1: TFLite benchmark_model — CPU / GPU / NNAPI / QNN delegate"
BM="$WORKDIR/android_aarch64_benchmark_model"; [ -s "$BM" ] || curl -fsSL -o "$BM" "$BM_URL" || fail "download benchmark_model failed"
ash "mkdir -p $DEV_DIR" >/dev/null; $ADB push "$BM" "$DEV_DIR/benchmark_model" >/dev/null && ash "chmod +x $DEV_DIR/benchmark_model" && pass "benchmark_model pushed"
QNN_ON_DEV=0
if [ -n "$QNN_LIBS" ] && [ -f "$QNN_LIBS/libQnnTFLiteDelegate.so" ]; then
  ash "mkdir -p $DEV_DIR/qnn" >/dev/null
  for l in libQnnTFLiteDelegate.so libQnnHtp.so libQnnHtpV68Stub.so libQnnHtpPrepare.so libQnnSystem.so libQnnGpu.so libQnnCpu.so; do [ -f "$QNN_LIBS/$l" ] && $ADB push "$QNN_LIBS/$l" "$DEV_DIR/qnn/" >/dev/null; done
  [ -n "$QNN_SKEL" ] && [ -f "$QNN_SKEL/libQnnHtpV68Skel.so" ] && $ADB push "$QNN_SKEL/libQnnHtpV68Skel.so" "$DEV_DIR/qnn/" >/dev/null
  ash "ls $DEV_DIR/qnn/libQnnHtpV68Skel.so" 2>/dev/null | grep -q Skel && { pass "QNN Android libs + v68 skel pushed to $DEV_DIR/qnn"; QNN_ON_DEV=1; } || warn "QNN libs pushed but no libQnnHtpV68Skel.so (pass --qnn-skel <QAIRT>/lib/hexagon-v68/unsigned)"
fi
TFL=$(ls "$MODELS_DIR"/*_saved_model/*_float32.tflite "$MODELS_DIR"/*_saved_model/*_full_integer_quant.tflite 2>/dev/null); [ -n "$TFL" ] || warn "No .tflite under $MODELS_DIR — run export_host.sh on the PC first"
run_bm() { # label file flags
  local label="$1" f="$2" flags="$3" m; m=$(basename "$f" .tflite | sed 's/_.*//')
  local OUT; OUT=$(ash "cd $DEV_DIR && LD_LIBRARY_PATH=$DEV_DIR/qnn ADSP_LIBRARY_PATH=$DEV_DIR/qnn ./benchmark_model --graph=$DEV_DIR/$(basename "$f") --num_runs=50 --warmup_runs=5 $flags" 2>&1)
  echo "$OUT" > "$LOGDIR/${label}_$(basename "$f" .tflite).log"
  local avg init first; avg=$(echo "$OUT" | grep -oE "Inference \(avg\): [0-9.]+" | grep -oE "[0-9.]+$"); init=$(echo "$OUT" | grep -oE "Init: [0-9.]+" | grep -oE "[0-9.]+"); first=$(echo "$OUT" | grep -oE "First inference: [0-9.]+" | grep -oE "[0-9.]+")
  if [ -n "$avg" ]; then local ms fps; ms=$(awk "BEGIN{printf \"%.2f\", $avg/1000}"); fps=$(awk "BEGIN{printf \"%.2f\", 1000000/$avg}")
    pass "[$label/$(basename "$f")] ${ms}ms avg (${fps} FPS) | init $(awk "BEGIN{printf \"%.1f\", ${init:-0}/1000}")ms | first $(awk "BEGIN{printf \"%.1f\", ${first:-0}/1000}")ms $(echo "$OUT" | grep -oE '(completely|partially) executed by the delegate[^.]*|not be executed by the delegate' | head -1 | sed 's/^/| /')"
    { echo "$label,$m,steady_avg,ms,$ms"; echo "$label,$m,video,fps,$fps"; echo "$label,$m,first_infer,ms,$(awk "BEGIN{printf \"%.1f\", ${first:-0}/1000}")"; } >> "$BENCH_CSV"
  else warn "[$label/$(basename "$f")] failed: $(echo "$OUT" | grep -iE 'error|fail|not found' | head -1 | cut -c1-160)"; fi
}
for f in $TFL; do
  $ADB push "$f" "$DEV_DIR/" >/dev/null || { warn "push failed: $f"; continue; }; isint=0; [[ "$f" == *integer_quant* ]] && isint=1
  run_bm "android_tflite_cpu" "$f" "--num_threads=4 --use_xnnpack=true"
  run_bm "android_tflite_gpu" "$f" "--use_gpu=true --gpu_precision_loss_allowed=true"
  for acc in qti-dsp qti-gpu qti-default; do [ -n "$NNHAL" ] && run_bm "android_tflite_nnapi_${acc}" "$f" "--use_nnapi=true --nnapi_accelerator_name=$acc"; done
  [ -z "$NNHAL" ] && run_bm "android_tflite_nnapi_reference" "$f" "--use_nnapi=true"
  if [ "$QNN_ON_DEV" -eq 1 ]; then
    run_bm "android_tflite_qnn_htp" "$f" "--external_delegate_path=$DEV_DIR/qnn/libQnnTFLiteDelegate.so --external_delegate_options=backend_type:htp;library_path:$DEV_DIR/qnn/libQnnHtp.so;skel_library_dir:$DEV_DIR/qnn;htp_precision:$([ $isint -eq 1 ] && echo 0 || echo 1);htp_performance_mode:2"
    run_bm "android_tflite_qnn_gpu" "$f" "--external_delegate_path=$DEV_DIR/qnn/libQnnTFLiteDelegate.so --external_delegate_options=backend_type:gpu;library_path:$DEV_DIR/qnn/libQnnGpu.so"
  fi
done

section "STEP 2: qnn-net-run (informational)"
if [ -n "$QNN_LIBS" ] && [ -x "$QNN_LIBS/../../bin/aarch64-android/qnn-net-run" ]; then
  $ADB push "$QNN_LIBS/../../bin/aarch64-android/qnn-net-run" "$DEV_DIR/qnn/" >/dev/null; ash "chmod +x $DEV_DIR/qnn/qnn-net-run"
  info "qnn-net-run pushed. To benchmark a context binary: extract it from <model>_qnn.onnx (the ONNX wrapper's EPContext node holds the blob) or generate one with qnn-context-binary-generator, then:"
  info "  adb shell 'cd $DEV_DIR/qnn && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./qnn-net-run --backend libQnnHtp.so --retrieve_context model.bin --input_list list.txt --perf_profile burst --num_inferences 100 --profiling_level basic'"
else info "qnn-net-run not available (needs the QAIRT SDK zip: bin/aarch64-android/qnn-net-run) — skipped"; fi

section "SUMMARY"
echo -e "${GREEN}Passed: $(grep -c '^\[PASS\]' "$RESULTS_FILE")${NC}  ${YELLOW}Warnings: $(grep -c '^\[WARN\]' "$RESULTS_FILE")${NC}  ${RED}Failed: $(grep -c '^\[FAIL\]' "$RESULTS_FILE")${NC}\n"; cat "$RESULTS_FILE"
echo -e "\n${BLUE}  INFERENCE-ONLY (benchmark_model; no pre/post-processing, no NMS)${NC}"; printf "%-36s %-10s %12s %12s\n" "BACKEND" "MODEL" "MS" "FPS"
grep ",steady_avg,ms," "$BENCH_CSV" | sort -t, -k2,2 -k1,1 | while IFS=, read -r b m s met v; do fps=$(grep "^$b,$m,video,fps," "$BENCH_CSV" | cut -d, -f5); printf "%-36s %-10s %12s %12s\n" "$b" "$m" "$v" "${fps:-}"; done
echo -e "\nCSV: $BENCH_CSV | logs: $LOGDIR"
