#!/usr/bin/env python3
import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt


def load_qoi_rows(path: Path, dt: float):
    rows = []
    header = None
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            # Skip partial rows while file is being appended.
            if header is not None and len(row) != len(header):
                continue
            try:
                vals = [float(x) for x in row]
            except ValueError:
                continue
            tstep = int(vals[0])
            rows.append((tstep * dt, vals[1:]))
    return header, rows


def build_plot(rows_coarse, rows_fine, qoi_names, output_path: Path, title: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True)
    axes = axes.ravel()

    tc = [t for t, _ in rows_coarse]
    tf = [t for t, _ in rows_fine]

    for i, q in enumerate(qoi_names):
        ax = axes[i]
        yc = [vals[i] for _, vals in rows_coarse]
        yf = [vals[i] for _, vals in rows_fine]

        ax.plot(tc, yc, label="dt=5.0e-05 (0.0001)", linewidth=1.6)
        ax.plot(tf, yf, label="dt=1.25e-05 (0.00005)", linewidth=1.6)
        ax.set_title(q)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Physical time (s)")
        ax.set_ylabel("QoI value")
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot long-run QoI comparison")
    parser.add_argument("--base-dir", default=".", help="Directory containing results_*_long folders")
    parser.add_argument("--coarse", default="results_0.0001_long/Extracted/kernel_qoi.csv")
    parser.add_argument("--fine", default="results_0.00005_long/Extracted/kernel_qoi.csv")
    parser.add_argument("--dt-coarse", type=float, default=5.0e-05)
    parser.add_argument("--dt-fine", type=float, default=1.25e-05)
    parser.add_argument("--out", default="results_qoi_comparison_long_current.png")
    parser.add_argument("--watch", action="store_true", help="Replot repeatedly while files grow")
    parser.add_argument("--interval", type=float, default=20.0, help="Refresh interval in seconds for --watch")
    args = parser.parse_args()

    base = Path(args.base_dir)
    coarse_path = base / args.coarse
    fine_path = base / args.fine
    out_path = base / args.out

    while True:
        hc, rc = load_qoi_rows(coarse_path, args.dt_coarse)
        hf, rf = load_qoi_rows(fine_path, args.dt_fine)

        if hc is None or hf is None:
            raise RuntimeError("Missing CSV header in one or both files")

        qoi_names = hc[1:]
        if hf[1:] != qoi_names:
            raise RuntimeError("QoI columns differ between files")

        build_plot(
            rows_coarse=rc,
            rows_fine=rf,
            qoi_names=qoi_names,
            output_path=out_path,
            title=f"QoI Comparison (coarse rows={len(rc)}, fine rows={len(rf)})",
        )

        print(f"Wrote {out_path} | coarse rows={len(rc)} | fine rows={len(rf)}")

        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
