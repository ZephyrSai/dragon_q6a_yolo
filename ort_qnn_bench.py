#!/usr/bin/env python3
"""
ort_qnn_bench.py — raw ONNX Runtime benchmark on the Dragon Q6A / QCS6490
through the QNN Execution Provider (Hexagon HTP NPU, Adreno GPU, or the
QNN CPU backend) or the plain CPU EP. No Ultralytics wrapper, no NMS.

Usage:
  python3 ort_qnn_bench.py --model model.onnx --backend <htp|gpu|cpu|ort-cpu>
      [--perf-mode burst|balanced|default|sustained_high_performance|power_saver]
      [--fp16] [--allow-fallback] [--video path.mp4] [--runs 15] [--video-max-frames 300]

Works with two kinds of ONNX files:
  - a plain FP32 export (yolo11n.onnx): the QNN EP compiles it for the
    chosen backend at session-create time (slow first time; HTP runs it in
    FP16 when --fp16 is given, otherwise it may fall back per-op)
  - an Ultralytics QNN export (yolo11n_qnn.onnx): an ONNX wrapper around a
    precompiled HTP v68 context binary — HTP only, near-instant load

Output (KEY=VALUE lines):
  ORT_VERSION / QNN_EP_VERSION / EP_DEVICES
  SESSION_CREATE_MS   - InferenceSession() incl. QNN graph compile for plain ONNX
  ACTUAL_PROVIDERS    - what ORT actually used (CPU-only here = fell back)
  FIRST_INFER_MS, STEADY_AVG_MS / MIN / MAX / P95
  VIDEO_FRAMES / VIDEO_AVG_FPS / VIDEO_AVG_MS  - decode + resize + run
"""

import argparse
import glob
import os
import statistics
import sys
import time

import numpy as np


def find_skel_dir():
    """Locate libQnnHtpV68Skel.so: env, onnxruntime_qnn wheel, system dirs."""
    cands = []
    env = os.environ.get("ADSP_LIBRARY_PATH", "")
    cands += [p for p in env.replace(";", ":").split(":") if p]
    try:
        import onnxruntime_qnn as q
        cands.append(os.path.dirname(q.get_library_path()))
        cands += glob.glob(os.path.join(os.path.dirname(q.__file__), "**"), recursive=True)
    except Exception:
        pass
    cands += ["/usr/lib/rfsa/adsp", "/usr/lib/dsp/cdsp", "/usr/lib/rfsa/adsp/cdsp", "/dsp", "/usr/lib"]
    for d in cands:
        if os.path.isdir(d) and glob.glob(os.path.join(d, "libQnnHtpV68Skel.so")):
            return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="htp", choices=["htp", "gpu", "cpu", "ort-cpu"])
    ap.add_argument("--perf-mode", default="burst")
    ap.add_argument("--fp16", action="store_true", help="enable_htp_fp16_precision for plain FP32 graphs on HTP")
    ap.add_argument("--allow-fallback", action="store_true", help="let unsupported ops run on the CPU EP instead of failing")
    ap.add_argument("--video", default=None)
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--video-max-frames", type=int, default=300)
    args = ap.parse_args()

    import onnxruntime as ort
    print(f"ORT_VERSION={ort.__version__}")
    print(f"MODEL={args.model}")
    print(f"BACKEND={args.backend}")

    so = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]
    provider_options = [{}]
    if args.backend != "ort-cpu":
        try:
            import onnxruntime_qnn as qnn_ep
        except Exception as e:
            print(f"ERROR=onnxruntime-qnn not importable: {e}")
            sys.exit(1)
        print(f"QNN_EP_VERSION={getattr(qnn_ep, '__version__', '?')}")
        skel = find_skel_dir()
        if skel and "ADSP_LIBRARY_PATH" not in os.environ:
            os.environ["ADSP_LIBRARY_PATH"] = skel
        print(f"ADSP_LIBRARY_PATH={os.environ.get('ADSP_LIBRARY_PATH', '<unset>')}")
        try:
            ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
        except Exception as e:
            if "already" not in str(e).lower():
                print(f"ERROR=register QNN EP failed: {str(e).splitlines()[0][:200]}")
                sys.exit(1)
        try:
            devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
            print(f"EP_DEVICES={[(d.ep_name, getattr(d.device, 'type', '?').__str__()) for d in devs]}")
        except Exception:
            pass
        opts = {"backend_type": args.backend, "htp_performance_mode": args.perf_mode, "htp_arch": "68"}
        if args.fp16:
            opts["enable_htp_fp16_precision"] = "1"
        providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
        provider_options = [opts, {}]
        if not args.allow_fallback:
            so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        print(f"QNN_OPTIONS={opts}")

    t0 = time.perf_counter()
    try:
        sess = ort.InferenceSession(args.model, so, providers=providers, provider_options=provider_options)
    except Exception as e:
        print(f"SESSION_ERROR={str(e).splitlines()[0][:300]}")
        sys.exit(1)
    print(f"SESSION_CREATE_MS={(time.perf_counter() - t0) * 1000:.1f}")
    print(f"ACTUAL_PROVIDERS={sess.get_providers()}")

    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    if len(shape) == 4 and shape[2] in (1, None):
        shape = [1, 3, 640, 640]
    dtype = np.float16 if "float16" in inp.type else np.float32
    nchw = shape[1] == 3
    print(f"INPUT_SHAPE={shape} {np.dtype(dtype).name}")
    dummy = np.random.rand(*shape).astype(dtype)

    t0 = time.perf_counter()
    try:
        sess.run(None, {inp.name: dummy})
    except Exception as e:
        print(f"FIRST_INFER_ERROR={str(e).splitlines()[0][:300]}")
        sys.exit(1)
    print(f"FIRST_INFER_MS={(time.perf_counter() - t0) * 1000:.2f}")

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: dummy})
        times.append((time.perf_counter() - t0) * 1000)
    ts = sorted(times)
    p95_idx = max(0, int(len(ts) * 0.95) - 1)
    print(f"STEADY_AVG_MS={statistics.mean(times):.2f}")
    print(f"STEADY_MIN_MS={min(times):.2f}")
    print(f"STEADY_MAX_MS={max(times):.2f}")
    print(f"STEADY_P95_MS={ts[p95_idx]:.2f}")

    if not args.video:
        return
    try:
        import cv2
    except Exception as e:
        print(f"VIDEO_ERROR=opencv not available: {e}")
        return
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"VIDEO_ERROR=could not open {args.video}")
        return
    h, w = (shape[2], shape[3]) if nchw else (shape[1], shape[2])
    n = 0
    ft = []
    t_start = time.perf_counter()
    while n < args.video_max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (rgb.transpose(2, 0, 1)[np.newaxis] if nchw else rgb[np.newaxis]).astype(dtype)
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        ft.append((time.perf_counter() - t0) * 1000)
        n += 1
    t_total = time.perf_counter() - t_start
    cap.release()
    if n:
        print(f"VIDEO_FRAMES={n}")
        print(f"VIDEO_TOTAL_S={t_total:.2f}")
        print(f"VIDEO_AVG_MS={statistics.mean(ft):.2f}")
        print(f"VIDEO_AVG_FPS={n / t_total:.2f}")


if __name__ == "__main__":
    main()
