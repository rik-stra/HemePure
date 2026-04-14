#!/usr/bin/env python3
"""Convert HemePure LF text output into per-timestep VTP files for ParaView.

Expected input row layout:
steps gridX gridY gridZ velX velY velZ pressure [mpirank]

The input is assumed to use the fixed LF header format with non-data metadata lines
followed by numeric rows. One .vtp file is written per timestep using sequence-index
naming.
"""

from __future__ import annotations

import argparse
from array import array
from pathlib import Path
from typing import List, Sequence, Tuple


OUTPUT_HEADER = ["gridX", "gridY", "gridZ", "velX", "velY", "velZ", "pressure"]


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Convert an LF text output file into one VTP file per timestep for ParaView."
		)
	)
	parser.add_argument("input_file", help="Path to input LF text file")
	parser.add_argument(
		"-o",
		"--output-dir",
		default=".",
		help="Directory where per-timestep VTP files are written (default: current directory)",
	)
	parser.add_argument(
		"-p",
		"--prefix",
		default="output",
		help="Output filename prefix (default: output)",
	)
	return parser


def convert_lf_to_vtp_series(input_path: Path, output_dir: Path, prefix: str) -> int:
	"""Stream LF rows and write one VTP file per timestep with low memory usage."""
	output_dir.mkdir(parents=True, exist_ok=True)

	current_timestep = None
	file_count = 0
	grid_x = array("d")
	grid_y = array("d")
	grid_z = array("d")
	vel_x = array("d")
	vel_y = array("d")
	vel_z = array("d")
	pressure = array("d")
	series_entries: List[Tuple[int, str]] = []

	def pack_float64(values: Sequence[float]) -> bytes:
		return array("d", values).tobytes()

	def pack_int32(values: Sequence[int]) -> bytes:
		return array("i", values).tobytes()

	def build_appended_blocks(num_points: int) -> List[bytes]:
		points = array("d")
		points_extend = points.extend
		for index in range(num_points):
			points_extend((grid_x[index], grid_y[index], grid_z[index]))
		connectivity = array("i", range(num_points))
		offsets = array("i", range(1, num_points + 1))
		return [
			pack_float64(grid_x),
			pack_float64(grid_y),
			pack_float64(grid_z),
			pack_float64(vel_x),
			pack_float64(vel_y),
			pack_float64(vel_z),
			pack_float64(pressure),
			pack_float64(points),
			pack_int32(connectivity),
			pack_int32(offsets),
		]

	def build_offsets(blocks: Sequence[bytes]) -> List[int]:
		offsets: List[int] = []
		current_offset = 0
		for block in blocks:
			offsets.append(current_offset)
			current_offset += 4 + len(block)
		return offsets

	def write_vtp_file(path: Path, num_points: int) -> None:
		blocks = build_appended_blocks(num_points)
		offsets = build_offsets(blocks)
		with path.open("wb") as handle:
			header = (
				'<?xml version="1.0"?>\n'
				'<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian" header_type="UInt32">\n'
				'  <PolyData>\n'
				f'    <Piece NumberOfPoints="{num_points}" NumberOfVerts="{num_points}" '
				f'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n'
				'      <PointData Scalars="pressure">\n'
				f'        <DataArray type="Float64" Name="gridX" format="appended" offset="{offsets[0]}"/>\n'
				f'        <DataArray type="Float64" Name="gridY" format="appended" offset="{offsets[1]}"/>\n'
				f'        <DataArray type="Float64" Name="gridZ" format="appended" offset="{offsets[2]}"/>\n'
				f'        <DataArray type="Float64" Name="velX" format="appended" offset="{offsets[3]}"/>\n'
				f'        <DataArray type="Float64" Name="velY" format="appended" offset="{offsets[4]}"/>\n'
				f'        <DataArray type="Float64" Name="velZ" format="appended" offset="{offsets[5]}"/>\n'
				f'        <DataArray type="Float64" Name="pressure" format="appended" offset="{offsets[6]}"/>\n'
				'      </PointData>\n'
				'      <Points>\n'
				f'        <DataArray type="Float64" NumberOfComponents="3" format="appended" offset="{offsets[7]}"/>\n'
				'      </Points>\n'
				'      <Verts>\n'
				f'        <DataArray type="Int32" Name="connectivity" format="appended" offset="{offsets[8]}"/>\n'
				f'        <DataArray type="Int32" Name="offsets" format="appended" offset="{offsets[9]}"/>\n'
				'      </Verts>\n'
				'    </Piece>\n'
				'  </PolyData>\n'
				'  <AppendedData encoding="raw">_'
			)
			handle.write(header.encode("utf-8"))
			for block in blocks:
				handle.write(len(block).to_bytes(4, byteorder="little", signed=False))
				handle.write(block)
			handle.write(b"\n  </AppendedData>\n</VTKFile>\n")

	def write_pvd_file(series: Sequence[Tuple[int, str]]) -> None:
		pvd_path = output_dir / f"{prefix}.pvd"
		with pvd_path.open("w", encoding="utf-8", newline="") as handle:
			handle.write('<?xml version="1.0"?>\n')
			handle.write(
				'<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
			)
			handle.write("  <Collection>\n")
			for timestep, filename in series:
				handle.write(
					f'    <DataSet timestep="{timestep}" group="" part="0" file="{filename}"/>\n'
				)
			handle.write("  </Collection>\n")
			handle.write("</VTKFile>\n")

	def flush_current_timestep() -> None:
		nonlocal grid_x, grid_y, grid_z, vel_x, vel_y, vel_z, pressure, file_count, series_entries
		num_points = len(grid_x)
		if num_points == 0:
			return
		output_path = output_dir / f"{prefix}_{file_count:04d}.vtp"
		write_vtp_file(output_path, num_points)
		series_entries.append((current_timestep if current_timestep is not None else file_count, output_path.name))
		grid_x = array("d")
		grid_y = array("d")
		grid_z = array("d")
		vel_x = array("d")
		vel_y = array("d")
		vel_z = array("d")
		pressure = array("d")
		file_count += 1

	with input_path.open("r", encoding="utf-8") as src:
		for line_number, raw_line in enumerate(src, start=1):
			stripped = raw_line.strip()
			if not stripped:
				continue

			parts = stripped.split()

			try:
				timestep_value = float(parts[0])
			except ValueError as exc:
				# Fixed-format header and metadata lines always begin with non-numeric tokens.
				continue

			if len(parts) < 8:
				raise ValueError(
					f"Malformed row at line {line_number}: expected at least 8 columns, got {len(parts)}"
				)

			timestep = int(timestep_value)

			if timestep != current_timestep:
				flush_current_timestep()
				current_timestep = timestep

			# Keep exactly: gridX, gridY, gridZ, velX, velY, velZ, pressure.
			grid_x.append(float(parts[1]))
			grid_y.append(float(parts[2]))
			grid_z.append(float(parts[3]))
			vel_x.append(float(parts[4]))
			vel_y.append(float(parts[5]))
			vel_z.append(float(parts[6]))
			pressure.append(float(parts[7]))

	flush_current_timestep()
	write_pvd_file(series_entries)

	if file_count == 0:
		raise ValueError(
			"No data rows found. Expected header lines plus rows like: "
			"step grid_x grid_y grid_z velocity(0) velocity(1) velocity(2) pressure"
		)

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

	file_count = convert_lf_to_vtp_series(input_path=input_path, output_dir=output_dir, prefix=prefix)

	print(
		f"Wrote {file_count} VTP files to {output_dir} from {input_path.name} "
		f"with point data arrays: {', '.join(OUTPUT_HEADER)}"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
