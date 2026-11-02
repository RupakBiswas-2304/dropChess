"""Exports a checkpoint to ONNX for the browser and checks it against PyTorch.

    python export.py                      # latest checkpoint -> ../web/public/models/dropchess.onnx
    python export.py --ckpt checkpoints/gen_030.pt --out /tmp/x.onnx --int8

Inputs/outputs: planes float32 [N, 31, 8, 8] -> policy [N, 4992] (raw logits,
unmasked), value [N, 1]. N is dynamic.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, planes):
        p, v = self.m(planes)
        return p, v.unsqueeze(1)


def fixture_planes(n: int = 100) -> np.ndarray:
    import dropchess

    xs = []
    for line in (HERE / "fixtures" / "encoding.jsonl").read_text().splitlines()[:n]:
        d = json.loads(line)
        g = dropchess.Game.from_fen(d["start"], d["variant"])
        for m in d["moves"]:
            g.apply_uci(m)
        xs.append(g.encode())
    return np.stack(xs).astype(np.float32)


def export(ckpt: str, out: Path, int8: bool = False):
    import onnxruntime as ort
    from run import load_model

    model, meta = load_model(ckpt, torch.device("cpu"))
    w = Wrapper(model).eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 31, 8, 8)
    torch.onnx.export(
        w, (dummy,), str(out), opset_version=17, dynamo=False,
        input_names=["planes"], output_names=["policy", "value"],
        dynamic_axes={"planes": {0: "N"}, "policy": {0: "N"}, "value": {0: "N"}},
    )
    x = fixture_planes()
    with torch.no_grad():
        p_ref, v_ref = w(torch.from_numpy(x))
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    p, v = sess.run(None, {"planes": x})
    dp, dv = np.abs(p - p_ref.numpy()).max(), np.abs(v - v_ref.numpy()).max()
    print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB) from gen {meta.get('gen')}; parity max diff policy {dp:.2e}, value {dv:.2e}")
    assert dp < 1e-4 and dv < 1e-4, "ONNX output differs from PyTorch"
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q = out.with_name(out.stem + ".int8.onnx")
        quantize_dynamic(str(out), str(q), weight_type=QuantType.QInt8)
        pq, vq = ort.InferenceSession(str(q), providers=["CPUExecutionProvider"]).run(None, {"planes": x})
        agree = (pq.argmax(1) == p.argmax(1)).mean()
        print(f"int8 copy {q} ({q.stat().st_size / 1e6:.1f} MB): top move agrees on {agree:.0%} of fixtures; "
              f"check its strength with evaluate.py before using it")


def main():
    from run import latest_checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="default: latest in checkpoints/")
    ap.add_argument("--out", default=str(HERE.parent / "web" / "public" / "models" / "dropchess.onnx"))
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()
    ckpt = args.ckpt or latest_checkpoint(HERE / "checkpoints")
    if ckpt is None:
        raise SystemExit("no checkpoint found")
    export(str(ckpt), Path(args.out), args.int8)


if __name__ == "__main__":
    main()
