"""법제처 OPEN API 호출."""

from __future__ import annotations

from typing import Any

import httpx

from .parser import strip_secrets

BASE_URL = "https://www.law.go.kr/DRF"


def _get_json(endpoint: str, oc: str, params: dict[str, str]) -> dict[str, Any]:
    response = httpx.get(
        f"{BASE_URL}/{endpoint}",
        params={"OC": oc, "type": "JSON", **params},
        timeout=30,
    )
    response.raise_for_status()
    return strip_secrets(response.json())


def find_mst(name: str, oc: str) -> str:
    """현행 법령 이름으로 법령일련번호를 찾는다."""
    data = _get_json("lawSearch.do", oc, {"target": "law", "query": name})
    laws = data["LawSearch"].get("law", [])
    if isinstance(laws, dict):
        laws = [laws]
    for law in laws:
        if law["법령명한글"] == name and law["현행연혁코드"] == "현행":
            return law["법령일련번호"]
    raise LookupError(f"현행 법령을 찾지 못했다: {name}")


def fetch_law(mst: str, oc: str) -> dict[str, Any]:
    return _get_json("lawService.do", oc, {"target": "law", "MST": mst})
