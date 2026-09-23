# Radxa Dragon Q6A (QCS6490) — the stack this suite needs, and the traps in it

`test_yolo_q6a.sh` benchmarks *on top of* an existing Qualcomm AI stack. This
file records what that stack has to be, how to build it, and the specific ways
it goes wrong. The QCS6490 is the hardest of the four boards this family of
suites covers, because it is the one where "the NPU is being used" is easiest to
believe and hardest to prove.

> **Verification status.** The *suite fixes* (one ONNX Runtime distribution,
> AutoUpdate off, reporting the provider that actually ran) were reproduced and
> fixed on AMD hardware, where the same bugs existed. The *Dragon Q6A install
> steps* below are assembled from Qualcomm and Radxa documentation and from
> community reports of working configurations, checked 23 Sep 2026, and have
> **not** been run on a Q6A by the author of this document. Treat §6 as the
> authority: it measures your board rather than trusting this page.

---

## 1. What has to be true

| Layer | What you need | Check |
|---|---|---|
| OS | Radxa OS / Ubuntu 24.04 (Noble) built for QCS6490 | `cat /etc/os-release` |
| Kernel | vendor kernel, `6.18.x-current-qcs6490` or the Radxa equivalent | `uname -r` |
| Firmware | `/lib/firmware/qcom/...` for the board | `ls /lib/firmware/qcom` |
| FastRPC | `/dev/fastrpc-cdsp` present and accessible | `ls -l /dev/fastrpc-*` |
| DSP skeleton | `libQnnHtpV68Skel.so` in the CDSP search path | §2.3 |
| QAIRT / QNN SDK | 2.46–2.47, **HTP arch v68** for QCS6490 | `qnn-net-run --version` |
| ONNX Runtime | **`onnxruntime-qnn`** only, 2.6.0 latest — never alongside `onnxruntime` | §3 |
| Model precision | INT8/UINT8 quantised — float models mostly fall back | §5 |

**v68 matters.** QCS6490's Hexagon is HTP **v68**. Skeletons and context binaries
built for v69/v73/v75 (8 Gen 1/2/3, Snapdragon X) will not load. Every artefact —
skeleton, QNN context binary, pre-compiled model — has to say v68. The skeleton
lives at `${QNN_SDK_ROOT}/lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so`.

The Hexagon architecture version does **not** track SoC model numbers in any
pattern you can extrapolate from — it has to come from the board's SoC id and the
SDK's own documentation. A wrong guess **builds cleanly and fails at load**,
which is why so many Q6A reports end in an opaque `qnn_open` error.

`onnxruntime-qnn` has its own 2.x version line, unrelated to ONNX Runtime's 1.x:

| | |
|---|---|
| Latest `onnxruntime-qnn` | 2.6.0 |
| Wheels for the Q6A | `manylinux_2_34_aarch64` — needs glibc ≥ 2.34 (Ubuntu 24.04 has 2.39) |
| Python | 3.11, 3.12, 3.13 or 3.14 only |

---

## 2. Procedure

### 2.1 Base system

```bash
sudo apt update
sudo apt install -y task-qualcomm embloader sdboot-is-embloader
sudo apt install -y fastrpc fastrpc-dev libcdsprpc1 radxa-firmware-qcs6490
```

### 2.2 Device node permissions

The FastRPC nodes are how anything reaches the DSP. Out of the box they are
often root-only, which produces a "permission denied" that surfaces much later
as an opaque QNN error:

```bash
sudo tee /etc/udev/rules.d/99-fastrpc.rules >/dev/null <<'EOF'
KERNEL=="fastrpc-*", MODE="0666"
SUBSYSTEM=="dma_heap", KERNEL=="system", MODE="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
ls -l /dev/fastrpc-*        # expect cdsp and adsp nodes, world-readable/writable
```

Confirm the DSP itself works before involving any ML framework:

```bash
fastrpc_test          # expect 3/3 tests to pass
```

If `fastrpc_test` fails, nothing above it can work. Stop here and fix the board.

### 2.3 QAIRT / QNN runtime and the DSP skeleton

Install the QAIRT (Qualcomm AI Runtime, formerly QNN SDK) release matching the
ONNX Runtime QNN EP you intend to use, then put the **v68** skeleton where the
DSP will find it:

```bash
sudo cp <QAIRT>/lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so /usr/lib/rfsa/adsp/cdsp/
export ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp/cdsp/
export LD_LIBRARY_PATH=/usr/lib/rfsa/adsp/cdsp/:$LD_LIBRARY_PATH
```

### 2.4 Python environment

```bash
python3 -m venv ~/yolo-q6a-test/venv && source ~/yolo-q6a-test/venv/bin/activate
pip install -U pip
pip install ultralytics "onnx>=1.12,<2" onnxslim

# ONNX Runtime: the QNN build, and NOTHING ELSE (see §3)
pip uninstall -y onnxruntime onnxruntime-qnn
pip install "onnxruntime-qnn>=2.4.0"        # needs Python >= 3.11, glibc >= 2.34
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# must contain QNNExecutionProvider
```

### 2.5 Run the suite

```bash
./test_yolo_q6a.sh
```

---

## 3. The trap that silently fakes your results

`onnxruntime` and `onnxruntime-qnn` are different distributions that unpack into
the **same `onnxruntime/` directory**. Installing both — which this suite used to
do in a single `pip install` line — means one shadows the other with no error at
all. You get whichever won, and every "NPU" number is really a CPU number.

It also happens without you asking: Ultralytics' exporter checks for a
distribution *literally named* `onnxruntime`, does not find it when only
`onnxruntime-qnn` is installed, and **AutoUpdate installs the plain CPU wheel
over your QNN build** mid-export.

On the AMD sibling suite this exact mechanism produced "GPU" ONNX results that
were entirely CPU. Defences now in this suite:

```bash
export YOLO_AUTOINSTALL=False          # Ultralytics may not install anything
pip list | grep -ci '^onnxruntime'     # must be exactly 1
```

plus `assert_ort_runtime` in the script, which re-checks after installs and
warns if `QNNExecutionProvider` disappears. Because AutoUpdate is off, `onnx`
and `onnxslim` are installed explicitly (§2.4).

---

## 4. The Qualcomm-specific trap: partial offload

Even when QNN initialises correctly, **the provider being first in
`get_providers()` does not mean your model ran on the NPU.** ONNX Runtime
partitions the graph: operators the HTP supports go to the DSP, everything else
silently runs on the CPU in the same session. A YOLO model with unsupported ops
can report `QNNExecutionProvider` while most of it executes on CPU.

So check the partitioning, not just the provider:

```python
import onnxruntime as ort
so = ort.SessionOptions()
so.log_severity_level = 1          # partitioning decisions are logged at INFO
sess = ort.InferenceSession("yolo11n.onnx", so,
                            providers=["QNNExecutionProvider"],
                            provider_options=[{"backend_path": "libQnnHtp.so",
                                               "htp_performance_mode": "burst"}])
print(sess.get_providers()[0])     # necessary, not sufficient
```

The log lines report how many nodes were assigned to QNN versus placed on CPU.
A model split into many small QNN partitions is usually *slower* than plain CPU,
because every partition boundary is a round trip to the DSP.

Cross-check with latency: if "NPU" is within noise of CPU, it is probably not
really on the NPU.

---

## 5. Quantisation is not optional

The HTP is an integer engine. Float32 YOLO models will either fail to offload or
offload so partially that the result is meaningless. For real numbers:

- quantise to INT8/UINT8 (per-tensor for HTP) with the AIMET/QAIRT quantiser or
  ONNX Runtime's static quantisation with a representative calibration set;
- export with **static shapes** (`dynamic=False`, fixed `imgsz`) — the HTP does
  not do dynamic shapes;
- consider pre-compiling to a **QNN context binary** for v68, which removes the
  per-run graph-compile cost that otherwise dominates first inference.

The suite reports cold-load, first-inference and steady-state separately so this
compile cost does not quietly inflate your throughput numbers.

---

## 6. First run on a new board — the short version

```bash
uname -r; cat /etc/os-release | head -2
ls -l /dev/fastrpc-*                     # nodes present and accessible
fastrpc_test                             # 3/3
ls /usr/lib/rfsa/adsp/cdsp/libQnnHtpV68Skel.so
echo "$ADSP_LIBRARY_PATH"
python3 -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
pip list | grep -i '^onnxruntime'        # exactly one line
python3 lib/accel_verify.py --list
./test_yolo_q6a.sh
```

---

## 7. Known failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `qnn_open failed, 0x80000600` | DSP-side `libc++.so.1` missing from the DSP search path; or host QNN version ≠ skeleton version; or the v68 skeleton is not in the CDSP directory | put the matching `libQnnHtpV68Skel.so` in `/usr/lib/rfsa/adsp/cdsp/`, symlink `libc++.so.1` and `libc++abi.so.1` into the board's DSP directory, set `ADSP_LIBRARY_PATH`, restart the CDSP remoteproc |
| Permission denied on `/dev/fastrpc-*` | udev rules not applied | §2.2, then reboot |
| `QNNExecutionProvider` absent | plain `onnxruntime` shadowing `onnxruntime-qnn`, or Python < 3.11 | §3 |
| NPU "works" but is no faster than CPU | partial offload (§4) or an unquantised model (§5) | check partitioning in the INFO log; quantise |
| Skeleton loads but inference fails | artefact built for v69/v73/v75 | rebuild for **v68** |
| Boot stalls | `qcom-apm` timeouts, WiFi driver | unrelated to the NPU; see Radxa's forum |

---

## 8. Verifying the silicon actually ran

`lib/accel_verify.py` reads kernel counters rather than trusting the runtime:

```bash
python3 lib/accel_verify.py --list
python3 lib/accel_verify.py --pid <pid> --seconds 5
```

Be aware of a real limitation on this platform: **the kernel exposes no busy
counter for the Hexagon HTP.** The tool reports the FastRPC nodes it can see
(so you know the path exists) and the Adreno GPU's `gpubusy` counter, but it
cannot prove HTP utilisation the way it can prove GPU utilisation on AMD or
Intel. On the Q6A the available evidence is:

1. `fastrpc_test` passing (the DSP path works at all),
2. the QNN EP's own partitioning log (§4 — how much of the graph went to the DSP),
3. QNN profiling (`profiling_level="detailed"`) for per-op placement and timing,
4. latency that is decisively better than the CPU row in `fair_compare.py`.

If all four agree, the NPU ran. If only the provider name says so, it did not.

---

## 9. Measuring fairly

`lib/fair_compare.py` compares every backend present, and `lib/fairness.py` holds
the methodology: rounds, medians of medians, reported spread, and a flag when the
spread exceeds 15%. It exists because naive measurement on the AMD sibling box
swung 46% on the same workload and produced a confidently wrong 1.8× where the
true figure was 1.1–1.3×.

On a passively cooled SoC like the Q6A this matters more, not less: sustained
load throttles, and the first run after boot is not the steady state.

```bash
python3 lib/fair_compare.py --model yolo11n --workdir ~/yolo-q6a-test
python3 lib/fair_compare.py --model yolo11n --mode interleaved
```

---

## 10. Files

- `INSTALL_DRAGON_Q6A.md` — this document
- `lib/accel_verify.py` — kernel-counter accelerator verification
- `lib/fairness.py`, `lib/fair_compare.py` — measurement methodology
- `test_yolo_q6a.sh` — the suite
- `ort_qnn_bench.py`, `tflite_bench.py`, `bench_harness.py` — per-stage harnesses
- `export_host.sh` — host-side model export/quantisation

---

## 11. Provenance of the version claims

The install steps here are assembled from vendor and community sources, checked
on **23 Sep 2026**, and have **not** been run on a Dragon Q6A.

| Claim | Source |
|---|---|
| QCS6490 is Hexagon HTP **v68**; skeleton at `lib/hexagon-v68/unsigned/`; arch version does not track SoC numbers and a wrong guess fails at load | Qualcomm QAIRT SDK documentation; [executorch #7356](https://github.com/pytorch/executorch/issues/7356) |
| `onnxruntime-qnn` 2.6.0, aarch64 `manylinux_2_34` wheels, Python ≥ 3.11 | [PyPI](https://pypi.org/project/onnxruntime-qnn/) |
| `qnn_open 0x80000600` causes and fix (DSP-side `libc++.so.1`, host/skel version mismatch, skeleton placement, `ADSP_LIBRARY_PATH`, CDSP remoteproc restart) | [Radxa forum thread](https://forum.radxa.com/t/radxa-dragon-q6a-dragon-q6a-ubuntu-24-04-qnn-htp-backend-fails-with-qnn-open-0x80000600/30850) — reported working with QAIRT 2.46.0 on kernel 6.18.2-current-qcs6490 |
| `task-qualcomm` / `fastrpc` packages, udev rules for `/dev/fastrpc-*`, `fastrpc_test` as the first check | Radxa documentation and community quickstart gists |
| Partial offload: the QNN EP partitions the graph and silently runs unsupported ops on CPU (§4) | same forum thread; consistent with ONNX Runtime's documented EP partitioning |
| The ONNX Runtime shadowing behaviour and Ultralytics AutoUpdate (§3) | reproduced and fixed on AMD hardware in [ryzen_yolo](https://github.com/ZephyrSai/ryzen_yolo) |

§6 prints what your board actually has. Where it disagrees with this table,
believe the board.
