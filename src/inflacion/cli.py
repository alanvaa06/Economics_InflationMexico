"""CLI mínima: ``inflacion refresh`` actualiza los datos desde el BIE."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from inflacion.config import settings
from inflacion.data.pipeline import refresh_inpc, refresh_inpc_quincenal_with_discovery


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(prog="inflacion")
    sub = parser.add_subparsers(dest="cmd", required=True)

    refresh = sub.add_parser("refresh", help="Descarga el INPC desde INEGI BIE.")
    refresh.add_argument(
        "--frequency",
        choices=("monthly", "quincenal"),
        default="monthly",
        help="Frecuencia a descargar (default: monthly).",
    )
    refresh.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Ruta de salida (.parquet o .xlsx). "
            "Default: data/RelevantInflation.parquet (mensual) o "
            "data/RelevantInflation_Q.parquet (quincenal)."
        ),
    )
    refresh.add_argument(
        "--recent-only",
        action="store_true",
        help="Trae solo los últimos puntos (más rápido; default: histórico completo).",
    )

    args = parser.parse_args(argv)

    if args.cmd == "refresh":
        settings.require_token()
        if args.frequency == "quincenal":
            out = args.output or settings.data_dir / "RelevantInflation_Q.parquet"
            refresh_inpc_quincenal_with_discovery(
                historic=not args.recent_only, out_path=out
            )
        else:
            out = args.output or settings.data_dir / "RelevantInflation.parquet"
            refresh_inpc(historic=not args.recent_only, out_path=out)
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
