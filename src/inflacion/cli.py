"""CLI mínima: ``inflacion refresh`` actualiza los datos desde el BIE."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from inflacion.config import settings
from inflacion.data.pipeline import refresh_inpc


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="inflacion")
    sub = parser.add_subparsers(dest="cmd", required=True)

    refresh = sub.add_parser("refresh", help="Descarga el INPC desde INEGI BIE.")
    refresh.add_argument(
        "--output",
        type=Path,
        default=settings.data_dir / "RelevantInflation.parquet",
        help="Ruta de salida (.parquet o .xlsx).",
    )
    refresh.add_argument(
        "--recent-only",
        action="store_true",
        help="Trae solo los últimos puntos (más rápido; default: histórico completo).",
    )

    args = parser.parse_args(argv)

    if args.cmd == "refresh":
        settings.require_token()
        refresh_inpc(historic=not args.recent_only, out_path=args.output)
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
