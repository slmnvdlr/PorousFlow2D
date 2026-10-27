from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import ConfigurationError, load_case
from .io import create_run_output_path, save_results
from .simulation import case_summary, prepare_case, run_case


class _Progress:
    def __init__(self) -> None:
        self.last_percent = -1

    def __call__(self, fraction: float, message: str) -> None:
        percent = int(100.0 * fraction)
        if percent >= self.last_percent + 5 or percent == 100:
            print(f"[{percent:3d}%] {message}", flush=True)
            self.last_percent = percent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="porous-fvm", description="Two-phase porous-media flow in a five-spot cell")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate input and build the mesh")
    validate.add_argument("case")
    mesh = sub.add_parser("mesh", help="build and save the flow-aligned mesh")
    mesh.add_argument("case")
    mesh.add_argument("-o", "--output", default="mesh.npz")
    run = sub.add_parser("run", help="run an implicit or explicit transport simulation")
    run.add_argument("case")
    run.add_argument("-o", "--output", default=None)
    sub.add_parser("gui", help="open the desktop interface")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        if args.command == "gui":
            from .gui import main as gui_main

            gui_main()
            return
        case = load_case(args.case)
        if args.command == "validate":
            prepared = prepare_case(case, _Progress())
            print(json.dumps(case_summary(case, prepared), ensure_ascii=False, indent=2))
        elif args.command == "mesh":
            prepared = prepare_case(case, _Progress())
            destination = Path(args.output).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination,
                nodes=prepared.mesh.nodes,
                cell_centers=prepared.mesh.cell_centers,
                cell_volumes=prepared.mesh.cell_volumes,
                permeability=prepared.material.permeability,
                porosity=prepared.material.porosity,
            )
            print(destination)
        elif args.command == "run":
            result = run_case(case, _Progress())
            output = (
                Path(args.output)
                if args.output
                else create_run_output_path(case.source_path, result)
            )
            print(save_results(result, output, source_project=case.source_path))
    except (ConfigurationError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
