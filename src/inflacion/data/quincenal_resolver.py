"""Resolver de IDs quincenales: sondea candidatos y persiste un sidecar JSON."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from inflacion.inegi.client import BIEClient, BIEError, classify_period

logger = logging.getLogger(__name__)


def resolve_quincenal_ids(
    client: BIEClient,
    candidates: Mapping[str, list[str]],
    *,
    sidecar_path: Path,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict[str, str]:
    """Para cada nombre, prueba los IDs candidatos y se queda con el primero quincenal.

    Persiste los resueltos en ``sidecar_path`` (JSON con metadata) y devuelve el
    mapping ``{nombre: id}``. Nombres sin candidato exitoso quedan fuera.
    """
    resolved: dict[str, dict[str, str]] = {}
    total = len(candidates)
    for index, (name, ids) in enumerate(candidates.items(), start=1):
        winner = _try_candidates(client, ids)
        if winner is not None:
            ind, sample_period = winner
            resolved[name] = {
                "id": ind,
                "period_format": "YYYY/MM/Qn",
                "last_period": sample_period,
            }
            logger.info("Quincenal '%s' resuelto a ID %s", name, ind)
        else:
            logger.warning("Quincenal '%s' no resuelto (todos los candidatos fallaron)", name)
        if progress_cb is not None:
            progress_cb(index, total, name)

    _write_sidecar(sidecar_path, resolved)
    return {name: meta["id"] for name, meta in resolved.items()}


def _try_candidates(client: BIEClient, ids: list[str]) -> tuple[str, str] | None:
    """Devuelve ``(id, último TIME_PERIOD)`` del primer ID que responda quincenal."""
    for ind in ids:
        url = client._url(ind, historic=False)
        try:
            payload = client._get(url)
        except BIEError as exc:
            logger.debug("Candidato %s falló: %s", ind, exc)
            continue
        if classify_period(payload) != "quincenal":
            logger.debug("Candidato %s no es quincenal", ind)
            continue
        try:
            sample = payload["Series"][0]["OBSERVATIONS"][-1]["TIME_PERIOD"]
        except (KeyError, IndexError):
            sample = ""
        return ind, sample
    return None


def _write_sidecar(path: Path, resolved: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "verified_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "ids": resolved,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
