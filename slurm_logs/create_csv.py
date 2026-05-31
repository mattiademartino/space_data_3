"""
parse_results.py

Scans a directory for .log files, extracts model names, best validation loss,
and PSNR, then writes a sorted CSV summary.

Usage:
    python parse_results.py                          # uses default LOG_DIR below
    python parse_results.py /path/to/logs            # pass directory as argument
    python parse_results.py /path/to/logs out.csv    # custom output path too
"""

import csv
import os
import re
import sys

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_LOG_DIR = "/cluster/home/mriestere/space_data_3"   # edit if needed
DEFAULT_OUTPUT  = "model_results.csv"
# ──────────────────────────────────────────────────────────────────────────────

# Patterns
RE_EXPERIMENT = re.compile(r"Experiment:\s+(\S+)")
RE_TRAIN_VAL  = re.compile(r"\[\S+\]\s+\d+/\d+\s+train\s+([\d.]+)\s+val\s+([\d.]+)")
RE_FINAL_VAL  = re.compile(r"->\s+Val MSE:\s+([\d.]+)\s+PSNR:\s+([\d.]+)\s+dB")
RE_PARAMS     = re.compile(r"Parameters:\s+([\d,]+)")


def parse_log_file(filepath: str) -> list[dict]:
    """Parse a single log file and return a list of model result dicts."""
    results = []
    current = {}

    with open(filepath, "r", errors="replace") as f:
        for line in f:
            line = line.strip()

            # New experiment block
            m = RE_EXPERIMENT.search(line)
            if m:
                if current:
                    results.append(current)
                current = {
                    "model":          m.group(1),
                    "params":         None,
                    "best_val_loss":  None,   # lowest val loss seen across epochs
                    "final_val_mse":  None,   # the reported Val MSE line
                    "psnr_db":        None,
                    "source_file":    os.path.basename(filepath),
                }
                continue

            if not current:
                continue

            # Parameter count
            m = RE_PARAMS.search(line)
            if m:
                current["params"] = int(m.group(1).replace(",", ""))
                continue

            # Per-epoch train/val line → track running best val loss
            m = RE_TRAIN_VAL.search(line)
            if m:
                val_loss = float(m.group(2))
                if current["best_val_loss"] is None or val_loss < current["best_val_loss"]:
                    current["best_val_loss"] = val_loss
                continue

            # Final summary line
            m = RE_FINAL_VAL.search(line)
            if m:
                current["final_val_mse"] = float(m.group(1))
                current["psnr_db"]       = float(m.group(2))
                continue

    if current:
        results.append(current)

    return results


def collect_all_results(log_dir: str) -> list[dict]:
    all_results = []
    for root, _, files in os.walk(log_dir):
        for fname in files:
            if fname.endswith(".log"):
                path = os.path.join(root, fname)
                parsed = parse_log_file(path)
                all_results.extend(parsed)
                print(f"  Parsed {len(parsed)} model(s) from {path}")
    return all_results


def write_csv(results: list[dict], output_path: str) -> None:
    # Sort by PSNR descending (best first); models without PSNR go to the bottom
    results.sort(key=lambda r: r["psnr_db"] if r["psnr_db"] is not None else -1,
                 reverse=True)

    fieldnames = [
        "rank",
        "model",
        "params",
        "best_val_loss",   # lowest val loss seen across all logged epochs
        "final_val_mse",   # val MSE reported in the summary line
        "psnr_db",
        "source_file",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(results, start=1):
            writer.writerow({
                "rank":           i,
                "model":          row["model"],
                "params":         row["params"],
                "best_val_loss":  f'{row["best_val_loss"]:.6f}' if row["best_val_loss"] is not None else "",
                "final_val_mse":  f'{row["final_val_mse"]:.6f}' if row["final_val_mse"] is not None else "",
                "psnr_db":        f'{row["psnr_db"]:.2f}'       if row["psnr_db"]       is not None else "",
                "source_file":    row["source_file"],
            })

    print(f"\nWrote {len(results)} rows → {output_path}")


def main():
    log_dir     = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG_DIR
    output_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT

    if not os.path.isdir(log_dir):
        print(f"Error: directory not found: {log_dir}")
        sys.exit(1)

    print(f"Scanning: {log_dir}")
    results = collect_all_results(log_dir)

    if not results:
        print("No model results found.")
        sys.exit(0)

    write_csv(results, output_path)


if __name__ == "__main__":
    main()