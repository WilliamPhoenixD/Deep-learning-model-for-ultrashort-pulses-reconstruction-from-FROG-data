#!/usr/bin/env python3
"""
Make a single multi-page PDF with ALL pulses:
- Keeps a GUI pop-up to choose the CSV path
- No plot windows (uses a non-interactive backend)
- One page per pulse: |E|, phase, intensity, and FROG heatmap
- By default, saves PDF next to the CSV as 'FROG_report.pdf'
- Optional: --out to change PDF path, --limit N to only include first N pulses,
            --summary <csv> to also write durations

Examples:
  python DatasetRead.py
  python DatasetRead.py --out "C:/somewhere/FROG_report.pdf" --limit 20 --summary durations.csv
"""

import argparse
import math
import os
import sys
import numpy as np

# Non-interactive backend to avoid windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ------------------------ UI: choose CSV via pop-up ------------------------ #

def choose_csv_gui():
    """Return a CSV path chosen via GUI; None if canceled or GUI not available."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Choose the pulses CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        root.destroy()
        return path if path else None
    except Exception:
        return None


# ------------------------------- Helpers ---------------------------------- #

def infer_N_from_row_length(L):
    """Given row length L, infer N from L = (N+1)^2."""
    sq = int(round(math.isqrt(L)))
    if sq * sq != L:
        sq = int(round(math.sqrt(L)))
    N = sq - 1
    if (N + 1) * (N + 1) != L or N <= 0:
        raise ValueError(f"Row length {L} is not a perfect square ((N+1)^2).")
    return N

def parse_row_to_tbp_E_T(row):
    """
    Row: [ TBP, Re[0..N-1], Im[0..N-1], T (N×N row-major) ]
    Returns (tbp: float, E: complex [N], T: float [N,N]).
    """
    L = row.size
    N = infer_N_from_row_length(L)
    tbp = float(row[0])
    re  = row[1 : 1 + N]
    im  = row[1 + N : 1 + 2 * N]
    T   = row[1 + 2 * N : ].reshape(N, N)
    E   = re + 1j * im
    return tbp, E, T

def read_rows_streaming(csv_path, delimiter=","):
    """Yield (idx, tbp, E, T) for each line in the CSV."""
    with open(csv_path, "r", encoding="utf-8") as f:
        idx = 0
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = np.fromstring(s, sep=delimiter)
            if row.size == 0:
                continue
            tbp, E, T = parse_row_to_tbp_E_T(row)
            yield idx, tbp, E, T
            idx += 1

def fwhm_index_from_intensity(I):
    """FWHM around global peak with linear interpolation between samples."""
    I = np.asarray(I, dtype=float)
    if I.size < 2:
        return np.nan
    Imax = float(I.max())
    if not np.isfinite(Imax) or Imax <= 0:
        return np.nan
    half = 0.5 * Imax
    k0 = int(np.argmax(I))

    # left crossing
    left = np.nan
    k = k0
    while k > 0 and I[k] >= half:
        k -= 1
    if k < k0 and I[k] < half:
        x0, y0 = k, I[k]
        x1, y1 = k+1, I[k+1]
        if y1 != y0:
            left = x0 + (half - y0) / (y1 - y0)

    # right crossing
    right = np.nan
    k = k0
    n = I.size
    while k < n-1 and I[k] >= half:
        k += 1
    if k > k0 and I[k] < half:
        x0, y0 = k-1, I[k-1]
        x1, y1 = k, I[k]
        if y1 != y0:
            right = x0 + (half - y0) / (y1 - y0)

    if np.isfinite(left) and np.isfinite(right):
        return float(right - left)
    return np.nan

def fwhm_from_rms_index(I):
    """RMS-equivalent FWHM (Gaussian: FWHM = 2.35482*sigma)."""
    I = np.asarray(I, dtype=float)
    S = I.sum()
    if S <= 0:
        return np.nan
    x = np.arange(I.size, dtype=float)
    mu = (x * I).sum() / S
    var = ((x - mu) ** 2 * I).sum() / S
    sigma = math.sqrt(max(var, 0.0))
    return 2.354820045 * sigma

def pulse_duration_index(E):
    """Return (duration_in_indices, method)."""
    I = np.abs(E) ** 2
    w = fwhm_index_from_intensity(I)
    if np.isfinite(w):
        return w, "FWHM"
    return fwhm_from_rms_index(I), "RMS"


# ------------------------------- Plotting ---------------------------------- #

def make_page_figure(tbp, E, T, duration, method, idx=None):
    """
    Build one Figure for the PDF:
      - 2×2 grid: |E| (top-left), phase (top-right), intensity (mid-right),
        metadata (mid-left), and FROG heatmap spanning bottom row.
    """
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait (inches)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.4], hspace=0.35, wspace=0.25)

    n = E.size
    x = np.arange(n)
    mag = np.abs(E)
    phase = np.unwrap(np.angle(E))
    I = mag**2

    # |E|
    ax_mag = fig.add_subplot(gs[0, 0])
    ax_mag.plot(x, mag)
    ax_mag.set_title("|E| (initial pulse)")
    ax_mag.set_xlabel("index"); ax_mag.set_ylabel("|E|")

    # Phase
    ax_phase = fig.add_subplot(gs[0, 1])
    ax_phase.plot(x, phase)
    ax_phase.set_title("Phase(E) (unwrapped)")
    ax_phase.set_xlabel("index"); ax_phase.set_ylabel("phase [rad]")

    # Metadata panel
    ax_meta = fig.add_subplot(gs[1, 0])
    ax_meta.axis("off")
    lines = []
    if idx is not None:
        lines.append(f"Pulse index: {idx}")
    lines.append(f"TBP: {tbp:.2f}")
    if np.isfinite(duration):
        lines.append(f"Duration (index units): {duration:.3f}  ({method})")
    else:
        lines.append("Duration (index units): n/a")
    ax_meta.text(0.0, 0.9, "\n".join(lines), va="top", ha="left", fontsize=11)

    # Intensity
    ax_int = fig.add_subplot(gs[1, 1])
    ax_int.plot(x, I)
    ax_int.set_title("Intensity |E|²")
    ax_int.set_xlabel("index"); ax_int.set_ylabel("intensity")

    # FROG heatmap (raw values, no normalization)
    ax_frog = fig.add_subplot(gs[2, :])
    T = np.asarray(T, dtype=float)
    im = ax_frog.imshow(T, origin="lower", aspect="auto")
    ax_frog.set_title("FROG trace (raw)")
    ax_frog.set_xlabel("delay index (τ)")
    ax_frog.set_ylabel("frequency index (ω)")
    cbar = fig.colorbar(im, ax=ax_frog, fraction=0.046, pad=0.04)
    cbar.set_label("intensity")


    fig.suptitle(f"Pulse {idx}" if idx is not None else "Pulse", fontsize=14, y=0.995)
    return fig


# --------------------------------- Main ----------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV path (if omitted, a file chooser pops up).")
    ap.add_argument("--delimiter", default=",", help="CSV delimiter (default: ,)")
    ap.add_argument("--out", help="Output PDF path (default: next to CSV as FROG_report.pdf).")
    ap.add_argument("--limit", type=int, default=None, help="Only include the first N pulses.")
    ap.add_argument("--summary", help="Also write durations to this CSV path (optional).")
    args = ap.parse_args()

    # Choose CSV: prefer arg; else GUI; else exit
    csv_path = args.csv or choose_csv_gui()
    if not csv_path:
        print("No CSV selected.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(csv_path):
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Decide output PDF path
    if args.out:
        out_pdf = args.out
    else:
        base_dir = os.path.dirname(os.path.abspath(csv_path))
        out_pdf = os.path.join(base_dir, "FROG_report.pdf")

    # Ensure output folder exists
    out_dir = os.path.dirname(os.path.abspath(out_pdf))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"Reading: {csv_path}")
    print(f"Saving PDF to: {out_pdf}")

    total = 0
    durations_rows = []  # (index, tbp, duration, method)

    with PdfPages(out_pdf) as pdf:
        for idx, tbp, E, T in read_rows_streaming(csv_path, delimiter=args.delimiter):
            dur, method = pulse_duration_index(E)
            durations_rows.append((idx, tbp, float(dur), method))
            fig = make_page_figure(tbp, E, T, dur, method, idx=idx)
            pdf.savefig(fig)
            plt.close(fig)

            total += 1
            if args.limit is not None and total >= args.limit:
                break

        # Add a summary page
        if durations_rows:
            durs = np.array([r[2] for r in durations_rows if np.isfinite(r[2])])
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_subplot(111); ax.axis("off")
            lines = [f"Total pulses: {len(durations_rows)}"]
            if durs.size:
                lines += [
                    "Duration stats (index units):",
                    f"  mean = {durs.mean():.3f}",
                    f"  min  = {durs.min():.3f}",
                    f"  max  = {durs.max():.3f}",
                ]
            ax.text(0.05, 0.95, "\n".join(lines), va="top", ha="left", fontsize=12)
            fig.suptitle("Summary", fontsize=14, y=0.99)
            pdf.savefig(fig)
            plt.close(fig)

    # Optional duration CSV
    if args.summary:
        import csv as _csv
        with open(args.summary, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["index", "tbp", "duration_index", "method"])
            w.writerows(durations_rows)
        print(f"Wrote durations → {os.path.abspath(args.summary)}")

    print(f"Done. Pulses processed: {total}")

if __name__ == "__main__":
    main()
