#!/usr/bin/env python3
"""Convert HemePure LF text output into per-timestep CSV files for ParaView.

Expected input row layout:
steps gridX gridY gridZ velX velY velZ pressure [mpirank]

The first two lines are skipped, then rows are grouped by timestep (first column).
One CSV file is written per timestep using sequence-index naming.
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List


OUTPUT_HEADER = ["gridX", "gridY", "gridZ", "velX", "velY", "velZ", "pressure"]


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Convert an LF text output file into one CSV per timestep for ParaView."
		)
	)
	parser.add_argument("input_file", help="Path to input LF text file")
	parser.add_argument(
		"-o",
		"--output-dir",
		default=".",
		help="Directory where per-timestep CSV files are written (default: current directory)",
	)
	parser.add_argument(
		"-p",
		"--prefix",
		default="output",
		help="Output filename prefix (default: output)",
	)
	parser.add_argument(
		"--skip-rows",
		type=int,
		default=2,
		help="Number of lines to skip at start of input file (default: 2)",
	)
	return parser


def parse_lf_file(input_path: Path, skip_rows: int) -> Dict[int, List[List[str]]]:
	"""Parse LF text file and return ordered timestep groups with selected fields."""
	if skip_rows < 0:
		raise ValueError("skip_rows must be >= 0")

	grouped_rows: "OrderedDict[int, List[List[str]]]" = OrderedDict()

	with input_path.open("r", encoding="utf-8") as handle:
		for line_number, raw_line in enumerate(handle, start=1):
			if line_number <= skip_rows:
				continue

			stripped = raw_line.strip()
			if not stripped:
				continue

			parts = stripped.split()
			if len(parts) < 8:
				raise ValueError(
					f"Malformed row at line {line_number}: expected at least 8 columns, got {len(parts)}"
				)

			try:
				timestep = int(float(parts[0]))
			except ValueError as exc:
				raise ValueError(
					f"Invalid timestep value at line {line_number}: {parts[0]}"
				) from exc

			# Keep exactly: gridX, gridY, gridZ, velX, velY, velZ, pressure
			values = parts[1:8]
			grouped_rows.setdefault(timestep, []).append(values)

	if not grouped_rows:
		raise ValueError("No data rows found after skipping header lines")

	return grouped_rows


def write_timestep_csvs(
	grouped_rows: Dict[int, List[List[str]]],
	output_dir: Path,
	prefix: str,
) -> int:
	"""Write one CSV per timestep using sequence indices and return file count."""
	output_dir.mkdir(parents=True, exist_ok=True)

	file_count = 0
	for index, timestep in enumerate(grouped_rows):
		output_path = output_dir / f"{prefix}_{index:04d}.csv"
		with output_path.open("w", newline="", encoding="utf-8") as handle:
			writer = csv.writer(handle)
			writer.writerow(OUTPUT_HEADER)
			writer.writerows(grouped_rows[timestep])
		file_count += 1

	return file_count


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()

	input_path = Path(args.input_file)
	if not input_path.is_file():
		parser.error(f"Input file not found: {input_path}")

	output_dir = Path(args.output_dir)
	prefix = args.prefix.strip()
	if not prefix:
		parser.error("Output prefix must not be empty")

	grouped_rows = parse_lf_file(input_path=input_path, skip_rows=args.skip_rows)
	file_count = write_timestep_csvs(grouped_rows=grouped_rows, output_dir=output_dir, prefix=prefix)

	print(
		f"Wrote {file_count} CSV files to {output_dir} from {input_path.name} "
		f"using header: {', '.join(OUTPUT_HEADER)}"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
