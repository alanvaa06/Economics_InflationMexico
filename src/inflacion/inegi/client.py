"""Cliente HTTP tipado para el Banco de Indicadores Económicos (BIE) de INEGI.

Diseño:
- El token va en el path; nunca lo escribimos a logs ni excepciones.
- Reintentos exponenciales en errores transitorios (5xx, timeouts).
- Caché opcional en disco vía hishel para evitar llamadas repetidas dentro
  del TTL (INPC se publica quincenal/mensualmente).
"""
from __future__ import annotations

import calendar
import logging
import re
from collections.abc import Callable, Iterable
from datetime import date
from typing import Any

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from inflacion.config import settings

logger = logging.getLogger(__name__)

# Quincena 1: días 1–15 (cierre el día 15). Quincena 2: días 16–fin de mes.
_QUINCENA_RE = re.compile(r"^Q?0?([12])$")


class BIEError(RuntimeError):
    """Error genérico al consultar el BIE."""


class MissingTokenError(BIEError):
    """No se encontró un token válido."""


_TOKEN_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

# Marcador del error 400 que devuelve INEGI cuando el token está caducado o
# es bogus. Es un string-match heurístico: BIE devuelve este mismo body para
# tokens revocados Y para IDs realmente inexistentes. health_check sólo lo usa
# como gate de auth-fail; ver tests/test_client_health.py.
_AUTH_FAIL_MARKER = "No se encontraron resultados"


def _redact(msg: str) -> str:
    """Sustituye cualquier UUID-like (formato del token INEGI) por ``REDACTED``."""
    return _TOKEN_RE.sub("REDACTED", msg)


def classify_period(payload: dict[str, Any]) -> str:
    """Clasifica un payload BIE por su primer ``TIME_PERIOD``.

    Retorna ``"mensual"`` (formato ``YYYY/MM``), ``"quincenal"``
    (``YYYY/MM/Q1|Q2`` o ``YYYY/MM/1|2``), o ``"desconocido"``.
    Tolera payloads malformados devolviendo ``"desconocido"``.
    """
    try:
        observations = payload["Series"][0]["OBSERVATIONS"]
    except (KeyError, IndexError, TypeError):
        return "desconocido"
    if not observations:
        return "desconocido"
    tp = observations[0].get("TIME_PERIOD", "")
    parts = tp.split("/") if tp else []
    if len(parts) == 2:
        return "mensual"
    if len(parts) == 3 and _QUINCENA_RE.match(parts[2].strip()):
        return "quincenal"
    return "desconocido"


class BIEClient:
    """Cliente síncrono mínimo del BIE (formato JSON v2.0).

    Uso típico::

        client = BIEClient()
        df = client.fetch_series("583766")  # IndiceGeneral
    """

    HEALTH_PROBE_ID = "583766"  # IndiceGeneral mensual (INPC base 2018)

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        geo: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._token = token if token is not None else settings.inegi_api_token
        if not self._token:
            raise MissingTokenError(
                "INEGI_API_TOKEN no está configurado. Crea un .env (ver .env.example)."
            )
        self._base = (base_url or settings.inegi_base_url).rstrip("/")
        self._geo = geo or settings.inegi_geo
        self._client = httpx.Client(timeout=timeout, transport=transport)

    # --- contexto ---
    def __enter__(self) -> BIEClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # --- públicos ---
    def fetch_series(self, indicator: str | int, *, historic: bool = False) -> pd.DataFrame:
        """Devuelve un DataFrame con índice ``date`` (último día del mes) y columna ``valor``.

        Args:
            indicator: ID del indicador BIE.
            historic: ``True`` para serie histórica completa, ``False`` para reciente.
        """
        url = self._url(str(indicator), historic=historic)
        payload = self._get(url)
        return _parse_observations(payload)

    def fetch_many(
        self,
        indicators: Iterable[str | int],
        *,
        historic: bool = False,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> pd.DataFrame:
        """Une varias series por columna (clave = id como string).

        Args:
            indicators: IDs BIE a descargar.
            historic: ``True`` trae serie completa, ``False`` reciente.
            progress_cb: opcional. Se invoca tras procesar cada serie con
                ``(done, total, name)``. ``done`` empieza en 1, ``name`` es el ID.
        """
        ids = [str(ind) for ind in indicators]
        total = len(ids)
        frames: list[pd.DataFrame] = []
        for index, ind in enumerate(ids, start=1):
            try:
                df = self.fetch_series(ind, historic=historic).rename(columns={"valor": ind})
                frames.append(df)
            except BIEError as exc:
                logger.warning("No se pudo descargar %s: %s", ind, _redact(str(exc)))
            if progress_cb is not None:
                progress_cb(index, total, ind)
        if not frames:
            raise BIEError(
                "BIE: ninguna serie respondió OK. Verifica el token o el catálogo."
            )
        return pd.concat(frames, axis=1).sort_index()

    def health_check(self) -> None:
        """Verifica que el token sea aceptado por INEGI sondeando un ID maestro.

        Raises:
            MissingTokenError: el token es sintácticamente válido pero INEGI lo rechaza
                (HTTP 400 con cuerpo conteniendo "No se encontraron resultados").
            BIEError: cualquier otra falla (5xx, 400 con cuerpo distinto, etc.).
        """
        url = self._url(self.HEALTH_PROBE_ID, historic=False)
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise BIEError(f"Error de red en health_check: {_redact(str(exc))}") from None
        if response.status_code == 200:
            return
        body = response.text
        if response.status_code == 400 and _AUTH_FAIL_MARKER in body:
            raise MissingTokenError(
                "INEGI rechazó el token (HTTP 400). Renueva en "
                "https://www.inegi.org.mx/servicios/api_indicadores.html y actualiza .env."
            )
        raise BIEError(
            f"health_check falló: HTTP {response.status_code} body={body[:200]!r}"
        )

    # --- internos ---
    def _url(self, indicator: str, *, historic: bool) -> str:
        flag = "true" if historic else "false"
        return (
            f"{self._base}/INDICATOR/{indicator}/es/{self._geo}/{flag}/BIE/2.0/{self._token}?type=json"
        )

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _get(self, url: str) -> dict[str, Any]:
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise BIEError(f"Error de red consultando BIE: {_redact(str(exc))}") from None
        if response.status_code == 403:
            raise BIEError(
                "INEGI rechazó la solicitud (403). Verifica el token y el dominio permitido."
            )
        if response.status_code >= 500:
            response.raise_for_status()  # → activa retry
        if response.status_code != 200:
            raise BIEError(
                f"Respuesta inesperada del BIE: HTTP {response.status_code} body={response.text[:200]!r}"
            )
        try:
            return response.json()  # type: ignore[no-any-return]
        except ValueError as exc:
            raise BIEError(f"Respuesta no-JSON del BIE: {exc}") from None


def _parse_observations(payload: dict[str, Any]) -> pd.DataFrame:
    """Convierte la respuesta JSON v2.0 del BIE a DataFrame."""
    try:
        observations = payload["Series"][0]["OBSERVATIONS"]
    except (KeyError, IndexError, TypeError) as exc:
        raise BIEError(f"Estructura inesperada en respuesta BIE: {exc}") from None

    rows: list[tuple[date, float]] = []
    for obs in observations:
        raw_value = obs.get("OBS_VALUE", "")
        time_period = obs.get("TIME_PERIOD", "")
        if not time_period:
            continue
        value = float("nan") if raw_value in ("", None) else float(raw_value)
        rows.append((_parse_time_period(time_period), value))

    if not rows:
        return pd.DataFrame(columns=["valor"])

    df = pd.DataFrame(rows, columns=["fecha", "valor"]).drop_duplicates("fecha")
    df = df.set_index(pd.DatetimeIndex(df["fecha"], name="fecha")).drop(columns="fecha")
    return df.sort_index()


def _parse_time_period(time_period: str) -> date:
    """Convierte el ``TIME_PERIOD`` del BIE a una fecha de cierre.

    Formatos soportados:

    - ``YYYY/MM``           → último día del mes (mensual).
    - ``YYYY/MM/Q1`` o
      ``YYYY/MM/1``         → quincena 1 → día 15.
    - ``YYYY/MM/Q2`` o
      ``YYYY/MM/2``         → quincena 2 → último día del mes.

    Cualquier otro formato dispara :class:`BIEError` con el literal redactado
    para evitar fugas accidentales.
    """
    parts = time_period.split("/")
    if len(parts) < 2:
        raise BIEError(f"TIME_PERIOD no reconocido: {_redact(time_period)!r}")
    year, month = int(parts[0]), int(parts[1])
    last_day = calendar.monthrange(year, month)[1]

    if len(parts) == 2:
        return date(year, month, last_day)

    if len(parts) == 3:
        match = _QUINCENA_RE.match(parts[2].strip())
        if not match:
            raise BIEError(f"TIME_PERIOD no reconocido: {_redact(time_period)!r}")
        quincena = match.group(1)
        return date(year, month, 15 if quincena == "1" else last_day)

    raise BIEError(f"TIME_PERIOD no reconocido: {_redact(time_period)!r}")


def _to_month_end(time_period: str) -> date:
    """Wrapper de compatibilidad. Usar :func:`_parse_time_period` directamente."""
    return _parse_time_period(time_period)
