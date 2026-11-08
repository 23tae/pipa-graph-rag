import hashlib

import pytest

from pipa_graph_rag.config import EXTRACTION_PROMPT_VERSION
from pipa_graph_rag.extraction.prompt import available_versions, load_prompt
from pipa_graph_rag.schema import ALLOWED_ENDPOINTS

# 만든 버전의 본문은 고치지 않는다. 이 값이 바뀌면 그 버전으로 만든 결과를 다시 만들 수 없다.
# 프롬프트를 바꾸려면 새 버전 파일을 추가하고 여기에 해시를 더한다.
PINNED_SHA256 = {
    "v2": "8359fad2d7c9547c2cf6b336422b5438595652b9c627562daf1d5b3eb3a33572",
    "v3": "302e396ab27c2285cac91a53f1a687acdab3b99b4863af70ccc31e915ab5bf01",
}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_every_version_file_is_pinned():
    assert set(available_versions()) == set(PINNED_SHA256)


@pytest.mark.parametrize("version", sorted(PINNED_SHA256))
def test_version_text_is_unchanged(version: str):
    assert sha256(load_prompt(version)) == PINNED_SHA256[version]


def test_latest_version_exists():
    assert EXTRACTION_PROMPT_VERSION in available_versions()


def test_unknown_version_names_available_ones():
    with pytest.raises(ValueError, match="v3"):
        load_prompt("v99")


def test_latest_prompt_lists_every_allowed_relation():
    prompt = load_prompt(EXTRACTION_PROMPT_VERSION)
    for kind in ALLOWED_ENDPOINTS:
        assert f"- {kind.value}:" in prompt
