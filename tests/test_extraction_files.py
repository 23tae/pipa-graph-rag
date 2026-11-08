import pytest

from pipa_graph_rag.extraction.pipeline import resolve_extraction_path


def make(directory, *names):
    for name in names:
        (directory / name).write_text("", encoding="utf-8")


def test_given_model_picks_that_models_file(tmp_path):
    make(tmp_path, "gpt-a-v3.jsonl", "gpt-b-v3.jsonl")
    path, model = resolve_extraction_path(tmp_path, "v3", "gpt-b", "gpt-a")
    assert (path.name, model) == ("gpt-b-v3.jsonl", "gpt-b")


def test_without_model_a_single_file_of_the_version_is_used_even_if_it_is_not_the_default(tmp_path):
    make(tmp_path, "gpt-a-v2.jsonl", "gpt-b-v3.jsonl")
    path, model = resolve_extraction_path(tmp_path, "v3", None, "gpt-default")
    assert (path.name, model) == ("gpt-b-v3.jsonl", "gpt-b")


def test_without_model_several_files_use_the_default_model(tmp_path):
    make(tmp_path, "gpt-a-v3.jsonl", "gpt-b-v3.jsonl")
    path, model = resolve_extraction_path(tmp_path, "v3", None, "gpt-b")
    assert (path.name, model) == ("gpt-b-v3.jsonl", "gpt-b")


def test_model_ids_with_dots_and_dashes_are_kept_whole(tmp_path):
    make(tmp_path, "gpt-6.1-terra-v3.jsonl")
    assert resolve_extraction_path(tmp_path, "v3", None, "x")[1] == "gpt-6.1-terra"


def test_missing_file_is_an_error_that_lists_the_available_files(tmp_path):
    make(tmp_path, "gpt-a-v3.jsonl", "gpt-b-v3.jsonl")
    with pytest.raises(FileNotFoundError, match="gpt-c-v3.jsonl.*gpt-a-v3.jsonl, gpt-b-v3.jsonl"):
        resolve_extraction_path(tmp_path, "v3", "gpt-c", "gpt-a")


def test_several_files_without_a_default_match_is_an_error(tmp_path):
    make(tmp_path, "gpt-a-v3.jsonl", "gpt-b-v3.jsonl")
    with pytest.raises(FileNotFoundError, match="gpt-default-v3.jsonl"):
        resolve_extraction_path(tmp_path, "v3", None, "gpt-default")


def test_no_file_at_all_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="있는 파일: 없음"):
        resolve_extraction_path(tmp_path, "v3", None, "gpt-a")
