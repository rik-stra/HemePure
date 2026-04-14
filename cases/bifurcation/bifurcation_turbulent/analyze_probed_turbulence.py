#!/usr/bin/env python3
"""Analyze probe extraction output for turbulence-like fluctuations.

This script reads HemeLB probe extraction files (probe*.dat) by invoking hemeXtract,
aggregates each probe to one time series (mean over selected sites per saved step), and
reports fluctuation metrics such as turbulence intensity.

Usage example:
  python analyze_probed_turbulence.py \
    --results-dir results_probed/Extracted \
    --input input.xml \
    --warmup-step 3000
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
import xml.etree.ElementTree as ET

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError("matplotlib is required for plotting. Install it in your Python environment.") from exc


@dataclass
class ProbeMetrics:
    name: str
    n_steps_total: int
    n_steps_used: int
    dt_physical: float
    mean_speed: float
    rms_speed_fluct: float
    ti_speed: float
    rms_u: float
    rms_v: float
    rms_w: float
    mean_pressure: float


@dataclass
class ProbeSeries:
    name: str
    steps: list[float]
    ux: list[float]
    uy: list[float]
    uz: list[float]
    p: list[float]
    speed: list[float]


@dataclass
class ProbeLocation:
    name: str
    x: float
    y: float
    z: float


@dataclass
class IoletGeometry:
    kind: str
    x: float
    y: float
    z: float
    nx: float
    ny: float
    nz: float
    radius: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze HemeLB probe outputs for turbulence-like fluctuations.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_probed/Extracted"),
        help="Directory containing probe*.dat files.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input.xml"),
        help="Input XML used for the run (for step_length conversion).",
    )
    parser.add_argument(
        "--hemextract",
        type=Path,
        default=Path("../../hemeXtract"),
        help="Path to hemeXtract executable.",
    )
    parser.add_argument(
        "--warmup-step",
        type=float,
        default=3000.0,
        help="Ignore steps below this lattice step for statistics.",
    )
    parser.add_argument(
        "--pattern",
        default="probe*.dat",
        help="Glob pattern for probe files.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("results_probed"),
        help="Directory where plots are written.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing plots.",
    )
    return parser.parse_args()


def get_step_length_seconds(input_xml: Path) -> float:
    tree = ET.parse(input_xml)
    root = tree.getroot()
    step = root.find("./simulation/step_length")
    if step is None:
        raise RuntimeError("Could not find simulation/step_length in input XML.")
    units = step.attrib.get("units", "")
    if units != "s":
        raise RuntimeError(f"Expected step_length units='s', got '{units}'.")
    return float(step.attrib["value"])


def parse_vec3(value: str) -> tuple[float, float, float]:
    txt = value.strip()
    if txt.startswith("(") and txt.endswith(")"):
        txt = txt[1:-1]
    parts = [p.strip() for p in txt.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3-vector, got: {value}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def get_probe_locations(input_xml: Path, probe_names: set[str]) -> list[ProbeLocation]:
    tree = ET.parse(input_xml)
    root = tree.getroot()

    locations: list[ProbeLocation] = []
    for po in root.findall("./properties/propertyoutput"):
        fname = po.attrib.get("file", "").strip()
        if not fname or fname not in probe_names:
            continue

        point = po.find("./geometry/point")
        if point is None:
            continue

        value = point.attrib.get("value", "").strip()
        if not value:
            continue

        try:
            x, y, z = parse_vec3(value)
        except ValueError:
            continue

        locations.append(ProbeLocation(name=fname, x=x, y=y, z=z))

    locations.sort(key=lambda p: p.name)
    return locations


def _norm3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def _normalize3(x: float, y: float, z: float) -> tuple[float, float, float]:
    n = _norm3(x, y, z)
    if n <= 0:
        return 0.0, 0.0, 1.0
    return x / n, y / n, z / n


def _cross3(ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> tuple[float, float, float]:
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


def get_iolet_geometry(input_xml: Path) -> list[IoletGeometry]:
    tree = ET.parse(input_xml)
    root = tree.getroot()

    voxel_node = root.find("./simulation/voxel_size")
    origin_node = root.find("./simulation/origin")
    if voxel_node is None or origin_node is None:
        return []

    voxel = float(voxel_node.attrib.get("value", "0"))
    ox, oy, oz = parse_vec3(origin_node.attrib.get("value", "(0,0,0)"))

    inlet_radius = None
    inlets = root.findall("./inlets/inlet")
    if inlets:
        rnode = inlets[0].find("./condition/radius")
        if rnode is not None:
            try:
                inlet_radius = float(rnode.attrib.get("value", "0"))
            except ValueError:
                inlet_radius = None

    iolets: list[IoletGeometry] = []

    for kind, tag in (("inlet", "./inlets/inlet"), ("outlet", "./outlets/outlet")):
        for node in root.findall(tag):
            pos_node = node.find("./position")
            nrm_node = node.find("./normal")
            if pos_node is None or nrm_node is None:
                continue

            try:
                px, py, pz = parse_vec3(pos_node.attrib.get("value", ""))
                nx, ny, nz = parse_vec3(nrm_node.attrib.get("value", ""))
            except ValueError:
                continue

            # Convert lattice coordinate iolet centers to physical units.
            x = ox + voxel * px
            y = oy + voxel * py
            z = oz + voxel * pz
            nx, ny, nz = _normalize3(nx, ny, nz)

            # Outlets don't carry radius in this input; use inlet radius fallback.
            radius = inlet_radius if inlet_radius is not None else 0.001
            iolets.append(IoletGeometry(kind=kind, x=x, y=y, z=z, nx=nx, ny=ny, nz=nz, radius=radius))

    return iolets


def get_target_spectral_exponent(input_xml: Path) -> float | None:
    tree = ET.parse(input_xml)
    root = tree.getroot()

    for inlet in root.findall("./inlets/inlet"):
        cond = inlet.find("./condition")
        if cond is None:
            continue
        if cond.attrib.get("subtype", "").strip().lower() != "turbulent":
            continue
        se = cond.find("./spectral_exponent")
        if se is None:
            continue
        try:
            return float(se.attrib.get("value", ""))
        except ValueError:
            return None

    return None


def run_heme_xtract(hemextract: Path, dat_file: Path) -> str:
    if not hemextract.exists():
        raise RuntimeError(f"hemeXtract not found at: {hemextract}")

    proc = subprocess.run(
        [str(hemextract), "-X", str(dat_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"hemeXtract failed for {dat_file} (exit {proc.returncode})\n{proc.stderr.strip()}"
        )
    return proc.stdout


def parse_extracted_text(text: str):
    # Aggregate rows per output step; each step has several nearby selected cells.
    by_step = defaultdict(lambda: {"ux": [], "uy": [], "uz": [], "p": []})

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("Reading") or line.startswith("Finished"):
            continue

        parts = re.split(r"\s+", line)
        if len(parts) < 8:
            continue

        try:
            step = float(parts[0])
            ux = float(parts[4])
            uy = float(parts[5])
            uz = float(parts[6])
            p = float(parts[7])
        except ValueError:
            continue

        by_step[step]["ux"].append(ux)
        by_step[step]["uy"].append(uy)
        by_step[step]["uz"].append(uz)
        by_step[step]["p"].append(p)

    if not by_step:
        return [], [], [], [], []

    steps = sorted(by_step.keys())
    ux_t = [mean(by_step[s]["ux"]) for s in steps]
    uy_t = [mean(by_step[s]["uy"]) for s in steps]
    uz_t = [mean(by_step[s]["uz"]) for s in steps]
    p_t = [mean(by_step[s]["p"]) for s in steps]
    speed_t = [math.sqrt(ux_t[i] ** 2 + uy_t[i] ** 2 + uz_t[i] ** 2) for i in range(len(steps))]

    return steps, ux_t, uy_t, uz_t, p_t, speed_t


def compute_metrics(name: str, step_length_s: float, warmup_step: float, steps, ux_t, uy_t, uz_t, p_t, speed_t) -> ProbeMetrics:
    if not steps:
        return ProbeMetrics(name, 0, 0, step_length_s, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    use_idx = [i for i, s in enumerate(steps) if s >= warmup_step]
    if not use_idx:
        use_idx = list(range(len(steps)))

    ux = [ux_t[i] for i in use_idx]
    uy = [uy_t[i] for i in use_idx]
    uz = [uz_t[i] for i in use_idx]
    pp = [p_t[i] for i in use_idx]
    ss = [speed_t[i] for i in use_idx]

    m_speed = mean(ss)
    rms_speed_fluct = pstdev(ss) if len(ss) > 1 else 0.0
    ti = rms_speed_fluct / m_speed if m_speed > 0 else float("nan")

    return ProbeMetrics(
        name=name,
        n_steps_total=len(steps),
        n_steps_used=len(use_idx),
        dt_physical=step_length_s,
        mean_speed=m_speed,
        rms_speed_fluct=rms_speed_fluct,
        ti_speed=ti,
        rms_u=pstdev(ux) if len(ux) > 1 else 0.0,
        rms_v=pstdev(uy) if len(uy) > 1 else 0.0,
        rms_w=pstdev(uz) if len(uz) > 1 else 0.0,
        mean_pressure=mean(pp),
    )


def classify_turbulence(avg_ti: float) -> str:
    if math.isnan(avg_ti):
        return "insufficient-data"
    if avg_ti < 0.02:
        return "mostly-steady"
    if avg_ti < 0.05:
        return "weakly-unsteady"
    if avg_ti < 0.10:
        return "transitional-like"
    return "strongly-unsteady"


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _variance(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    m = _mean(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)


def build_used_arrays(series: ProbeSeries, warmup_step: float):
    use_idx = [i for i, s in enumerate(series.steps) if s >= warmup_step]
    if not use_idx:
        use_idx = list(range(len(series.steps)))

    t_steps = [float(series.steps[i]) for i in use_idx]
    ux = [float(series.ux[i]) for i in use_idx]
    uy = [float(series.uy[i]) for i in use_idx]
    uz = [float(series.uz[i]) for i in use_idx]
    speed = [float(series.speed[i]) for i in use_idx]
    return t_steps, ux, uy, uz, speed


def plot_time_series(series_list: list[ProbeSeries], warmup_step: float, dt_step: float, out_path: Path) -> None:
    plt.figure(figsize=(10, 5.5))

    for s in series_list:
        t_steps, _ux, _uy, _uz, speed = build_used_arrays(s, warmup_step)
        if len(t_steps) == 0:
            continue
        time_s = [t * dt_step for t in t_steps]
        m_speed = _mean(speed)
        speed_fluct = [v - m_speed for v in speed]
        plt.plot(time_s, speed_fluct, linewidth=1.0, alpha=0.7, label=s.name.replace(".dat", ""))

    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Speed fluctuation |u|' [m/s]")
    plt.title("Probe Speed Fluctuation Time Series")
    plt.grid(True, alpha=0.3)
    if len(series_list) <= 12:
        plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _tke_spectrum_one_probe(ux: list[float], uy: list[float], uz: list[float], dt: float):
    # Remove mean to isolate fluctuations.
    mux = _mean(ux)
    muy = _mean(uy)
    muz = _mean(uz)
    up = [u - mux for u in ux]
    vp = [v - muy for v in uy]
    wp = [w - muz for w in uz]

    n = len(up)
    if n < 8:
        return None, None

    # Use Hann window for a smoother estimate.
    win = [0.5 * (1.0 - math.cos(2.0 * math.pi * i / (n - 1))) for i in range(n)]
    wnorm = sum(w * w for w in win)
    if wnorm <= 0:
        return None, None

    uw = [up[i] * win[i] for i in range(n)]
    vw = [vp[i] * win[i] for i in range(n)]
    ww = [wp[i] * win[i] for i in range(n)]

    # Real FFT bins via direct DFT (small probe time series => acceptable cost).
    kmax = n // 2
    freq = [k / (n * dt) for k in range(kmax + 1)]
    pxx_u = []
    pxx_v = []
    pxx_w = []

    for k in range(kmax + 1):
        ur = 0.0
        ui = 0.0
        vr = 0.0
        vi = 0.0
        wr = 0.0
        wi = 0.0
        for idx in range(n):
            ang = -2.0 * math.pi * k * idx / n
            c = math.cos(ang)
            s = math.sin(ang)
            ur += uw[idx] * c
            ui += uw[idx] * s
            vr += vw[idx] * c
            vi += vw[idx] * s
            wr += ww[idx] * c
            wi += ww[idx] * s

        pu = (ur * ur + ui * ui) / (wnorm / dt)
        pv = (vr * vr + vi * vi) / (wnorm / dt)
        pw = (wr * wr + wi * wi) / (wnorm / dt)

        # one-sided scaling except DC and Nyquist bins
        if k != 0 and k != kmax:
            pu *= 2.0
            pv *= 2.0
            pw *= 2.0

        pxx_u.append(pu)
        pxx_v.append(pv)
        pxx_w.append(pw)

    # TKE spectral density proxy: E(f) = 0.5*(Puu + Pvv + Pww)
    e_f = [0.5 * (pxx_u[i] + pxx_v[i] + pxx_w[i]) for i in range(len(freq))]
    return freq, e_f


def compute_mean_tke_spectrum(series_list: list[ProbeSeries], warmup_step: float, dt_step: float):
    spectra = []
    freq_ref = None

    for s in series_list:
        t_steps, ux, uy, uz, _speed = build_used_arrays(s, warmup_step)
        if len(t_steps) < 8:
            continue
        if len(t_steps) > 1:
            diffs = [t_steps[i + 1] - t_steps[i] for i in range(len(t_steps) - 1)]
            dt = _mean(diffs) * dt_step
        else:
            dt = dt_step
        freq, e_f = _tke_spectrum_one_probe(ux, uy, uz, dt)
        if freq is None:
            continue
        if freq_ref is None:
            freq_ref = freq
        if len(freq) == len(freq_ref):
            spectra.append(e_f)

    if not spectra or freq_ref is None:
        return None, None

    spec_mean = []
    for i in range(len(freq_ref)):
        spec_mean.append(sum(sp[i] for sp in spectra) / len(spectra))

    # Drop zero frequency for log-log plotting.
    freq_plot = []
    spec_plot = []
    for i in range(1, len(freq_ref)):
        if freq_ref[i] > 0 and spec_mean[i] > 0:
            freq_plot.append(freq_ref[i])
            spec_plot.append(spec_mean[i])

    if len(freq_plot) < 3:
        return None, None

    return freq_plot, spec_plot


def fit_loglog_slope(freq: list[float], spec: list[float], fmin: float, fmax: float) -> tuple[float | None, int]:
    xs = []
    ys = []
    for f, e in zip(freq, spec):
        if f >= fmin and f <= fmax and f > 0 and e > 0:
            xs.append(math.log(f))
            ys.append(math.log(e))

    n = len(xs)
    if n < 3:
        return None, n

    mx = _mean(xs)
    my = _mean(ys)
    varx = sum((x - mx) ** 2 for x in xs)
    if varx <= 0.0:
        return None, n
    covxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    slope = covxy / varx
    return slope, n


def plot_tke_spectrum(series_list: list[ProbeSeries], warmup_step: float, dt_step: float, out_path: Path) -> None:
    freq_plot, spec_plot = compute_mean_tke_spectrum(series_list, warmup_step, dt_step)
    if freq_plot is None or spec_plot is None:
        return

    plt.figure(figsize=(8.5, 5.2))
    plt.loglog(freq_plot, spec_plot, linewidth=2.0, label="Mean probe TKE spectrum")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("E(f) [m^2/s]")
    plt.title("TKE Spectrum (Probe-Averaged)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_probe_locations(locations: list[ProbeLocation], out_path: Path) -> None:
    if not locations:
        return

    xs = [p.x for p in locations]
    ys = [p.y for p in locations]
    zs = [p.z for p in locations]
    labels = [p.name.replace(".dat", "") for p in locations]

    cmap = plt.get_cmap("viridis", len(locations))
    colors = [cmap(i) for i in range(len(locations))]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    pairs = [
        (xs, ys, "x [m]", "y [m]", "XY"),
        (xs, zs, "x [m]", "z [m]", "XZ"),
        (ys, zs, "y [m]", "z [m]", "YZ"),
    ]

    for ax, (a, b, lx, ly, title) in zip(axes, pairs):
        ax.scatter(a, b, c=colors, s=48, edgecolors="black", linewidths=0.3)
        for i, lab in enumerate(labels):
            ax.text(a[i], b[i], f" {lab}", fontsize=7, va="center")
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Probe Locations")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _circle_in_plane(cx: float, cy: float, cz: float, nx: float, ny: float, nz: float, r: float, npts: int = 80):
    # Build orthonormal basis (u, v) perpendicular to plane normal n.
    if abs(nz) < 0.9:
        rx, ry, rz = 0.0, 0.0, 1.0
    else:
        rx, ry, rz = 1.0, 0.0, 0.0

    ux, uy, uz = _cross3(nx, ny, nz, rx, ry, rz)
    ux, uy, uz = _normalize3(ux, uy, uz)
    vx, vy, vz = _cross3(nx, ny, nz, ux, uy, uz)
    vx, vy, vz = _normalize3(vx, vy, vz)

    xs = []
    ys = []
    zs = []
    for i in range(npts + 1):
        t = 2.0 * math.pi * i / npts
        ct = math.cos(t)
        st = math.sin(t)
        xs.append(cx + r * (ct * ux + st * vx))
        ys.append(cy + r * (ct * uy + st * vy))
        zs.append(cz + r * (ct * uz + st * vz))
    return xs, ys, zs


def plot_probe_locations_3d(locations: list[ProbeLocation], iolets: list[IoletGeometry], out_path: Path) -> None:
    if not locations:
        return

    xs = [p.x for p in locations]
    ys = [p.y for p in locations]
    zs = [p.z for p in locations]
    labels = [p.name.replace(".dat", "") for p in locations]

    cmap = plt.get_cmap("viridis", len(locations))
    colors = [cmap(i) for i in range(len(locations))]

    fig = plt.figure(figsize=(8.2, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, c=colors, s=56, depthshade=True, edgecolors="black", linewidths=0.35)

    for i, label in enumerate(labels):
        ax.text(xs[i], ys[i], zs[i], f" {label}", fontsize=8)

    # A thin guide curve helps visually follow the probe order in space.
    ax.plot(xs, ys, zs, color="gray", linewidth=1.0, alpha=0.7)

    # Draw approximate channel contour from iolet cross-sections and centerlines.
    if iolets:
        inlet_nodes = [io for io in iolets if io.kind == "inlet"]
        outlet_nodes = [io for io in iolets if io.kind == "outlet"]
        if inlet_nodes and outlet_nodes:
            ix = sum(io.x for io in inlet_nodes) / len(inlet_nodes)
            iy = sum(io.y for io in inlet_nodes) / len(inlet_nodes)
            iz = sum(io.z for io in inlet_nodes) / len(inlet_nodes)
            ox = sum(io.x for io in outlet_nodes) / len(outlet_nodes)
            oy = sum(io.y for io in outlet_nodes) / len(outlet_nodes)
            oz = sum(io.z for io in outlet_nodes) / len(outlet_nodes)
            bx = 0.5 * (ix + ox)
            by = 0.5 * (iy + oy)
            bz = 0.5 * (iz + oz)

            for io in iolets:
                cxs, cys, czs = _circle_in_plane(io.x, io.y, io.z, io.nx, io.ny, io.nz, io.radius)
                ax.plot(cxs, cys, czs, color="tab:blue", linewidth=1.4, alpha=0.8)
                ax.plot([io.x, bx], [io.y, by], [io.z, bz], color="tab:blue", linewidth=1.1, alpha=0.5)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Probe Locations (3D) with Channel Contour")

    # Keep near-equal aspect so spacing is physically interpretable.
    xr = max(xs) - min(xs)
    yr = max(ys) - min(ys)
    zr = max(zs) - min(zs)
    if iolets:
        iox = [io.x for io in iolets]
        ioy = [io.y for io in iolets]
        ioz = [io.z for io in iolets]
        xr = max(xr, max(iox) - min(iox))
        yr = max(yr, max(ioy) - min(ioy))
        zr = max(zr, max(ioz) - min(ioz))
        xs.extend(iox)
        ys.extend(ioy)
        zs.extend(ioz)
    rmax = max(xr, yr, zr, 1e-12)
    cx = 0.5 * (max(xs) + min(xs))
    cy = 0.5 * (max(ys) + min(ys))
    cz = 0.5 * (max(zs) + min(zs))
    half = 0.5 * rmax
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    results_dir = args.results_dir.resolve()
    input_xml = args.input.resolve()
    hemextract = args.hemextract.resolve()

    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        return 2

    probe_files = sorted(results_dir.glob(args.pattern))
    if not probe_files:
        print(f"ERROR: no files matched {args.pattern} in {results_dir}", file=sys.stderr)
        return 2

    step_length_s = get_step_length_seconds(input_xml)

    metrics = []
    series_list: list[ProbeSeries] = []
    for pf in probe_files:
        text = run_heme_xtract(hemextract, pf)
        steps, ux_t, uy_t, uz_t, p_t, speed_t = parse_extracted_text(text)
        series_list.append(
            ProbeSeries(
                name=pf.name,
                steps=steps,
                ux=ux_t,
                uy=uy_t,
                uz=uz_t,
                p=p_t,
                speed=speed_t,
            )
        )
        m = compute_metrics(
            name=pf.name,
            step_length_s=step_length_s,
            warmup_step=args.warmup_step,
            steps=steps,
            ux_t=ux_t,
            uy_t=uy_t,
            uz_t=uz_t,
            p_t=p_t,
            speed_t=speed_t,
        )
        metrics.append(m)

    avg_ti = mean([m.ti_speed for m in metrics if not math.isnan(m.ti_speed)])
    label = classify_turbulence(avg_ti)

    print("Probe turbulence analysis")
    print(f"results_dir: {results_dir}")
    print(f"probe_files: {len(metrics)}")
    print(f"warmup_step: {args.warmup_step}")
    print(f"step_length: {step_length_s:.3e} s")
    print("")
    print("name         n_used  mean|u|[m/s]  rms(u')[m/s]  TI[-]    rms_u    rms_v    rms_w   mean_p")
    for m in metrics:
        print(
            f"{m.name:12s} {m.n_steps_used:6d}  "
            f"{m.mean_speed:12.6e} {m.rms_speed_fluct:12.6e} {m.ti_speed:7.4f} "
            f"{m.rms_u:8.3e} {m.rms_v:8.3e} {m.rms_w:8.3e} {m.mean_pressure:8.3e}"
        )

    print("")
    print(f"Average probe TI: {avg_ti:.4f}")
    print(f"Classification: {label}")
    print("Note: TI here is temporal fluctuation of probe-averaged speed, not a full turbulence model validation.")

    target_exp = get_target_spectral_exponent(input_xml)
    freq_plot, spec_plot = compute_mean_tke_spectrum(series_list, args.warmup_step, step_length_s)
    if freq_plot is not None and spec_plot is not None:
        fmin = freq_plot[max(1, int(0.15 * len(freq_plot)))]
        fmax = freq_plot[max(2, int(0.60 * len(freq_plot)))]
        slope, nfit = fit_loglog_slope(freq_plot, spec_plot, fmin, fmax)
        print("")
        print("Spectrum fit (log-log, mid-band)")
        print(f"fit_band_hz: [{fmin:.3e}, {fmax:.3e}]  points: {nfit}")
        if slope is not None:
            print(f"measured_slope: {slope:.4f}  (E(f) ~ f^{slope:.4f})")
            if target_exp is not None:
                target_slope = -target_exp
                print(f"target_slope:   {target_slope:.4f}  (from spectral_exponent={target_exp:.4f})")
                print(f"slope_error:    {slope - target_slope:+.4f}")
        else:
            print("measured_slope: unavailable (insufficient valid points)")

    if not args.no_plots:
        plot_dir = args.plot_dir.resolve()
        ts_path = plot_dir / "probe_speed_time_series.png"
        spec_path = plot_dir / "probe_tke_spectrum.png"
        loc_path = plot_dir / "probe_locations.png"
        loc3d_path = plot_dir / "probe_locations_3d.png"
        probe_locations = get_probe_locations(input_xml, {s.name for s in series_list})
        iolets = get_iolet_geometry(input_xml)
        plot_time_series(series_list, args.warmup_step, step_length_s, ts_path)
        plot_tke_spectrum(series_list, args.warmup_step, step_length_s, spec_path)
        plot_probe_locations(probe_locations, loc_path)
        plot_probe_locations_3d(probe_locations, iolets, loc3d_path)
        print("")
        print(f"Saved: {ts_path}")
        print(f"Saved: {spec_path}")
        if probe_locations:
            print(f"Saved: {loc_path}")
            print(f"Saved: {loc3d_path}")
        else:
            print("Note: no probe locations found in input XML.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
