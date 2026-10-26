"""Validation tests (SPEC 8.3/9) with fake LLM answers, good and bad ones."""

from __future__ import annotations

import pytest

from book2md.validate import (
    check_format_output,
    check_repair_output,
    count_header_like_tokens,
    diacritic_share,
    english_stopword_stats,
    fold_diacritics,
    normalize_nfc,
    sequence_similarity,
    strip_code_fences,
    strip_model_thinking,
    tokenize,
)


# ---- helpers --------------------------------------------------------------
def page(n: int, text: str) -> str:
    return f"<<<PAGE {n}>>>\n{text}"


POLISH_CHUNK = page(1, "Tamtej zimy zyli sobie w malej wiosce nad Bugiem.\nRano chlopaki czekali pod studnia i rozmawiali o zniwach.")
POLISH_REPAIRED = page(1, "Tamtej zimy żyli sobie w małej wiosce nad Bugiem.\nRano chłopaki czekali pod studnią i rozmawiali o żniwach.")
CONTEXT_BEFORE = " ".join(f"w{i}" for i in range(24))
CONTEXT_AFTER = " ".join(f"x{i}" for i in range(24))


# ---- tokenization / normalization ----------------------------------------
def test_tokenize_keeps_diacritics_and_casefolds():
    tokens = tokenize("Się żeby ŻÓŁĆ")
    assert tokens == ["się", "żeby", "żółć"]


def test_tokenize_strips_punctuation_and_markdown():
    assert tokenize("# Ładny, „tekst” — *woda* _brie_") == ["ładny", "tekst", "woda", "brie"]


def test_fold_diacritics_maps_polish_letters():
    # fold maps Polish letters to base forms but does NOT change case;
    # case is folded separately by tokenize()
    assert fold_diacritics("Się, ŻÓŁĆ ąę") == "Sie, ZOLC ae"


def test_normalize_nfc_idempotent():
    assert normalize_nfc("żółć") == normalize_nfc(normalize_nfc("żółć"))


def test_diacritic_share():
    assert diacritic_share("szkoda") == 0.0
    share = diacritic_share("żółć ala")  # 4 diacritics / 7 letters
    assert share == pytest.approx(4 / 7)


# ---- fence / thinking salvage --------------------------------------------
def test_strip_code_fences_removes_wrapping_pair_only():
    fenced = "```markdown\ntekst\n```"
    assert strip_code_fences(fenced) == "tekst"
    assert strip_code_fences("B\t\n<plain>") == "B\t\n<plain>"


def test_strip_model_thinking_removes_blocks():
    text = "<thinking>internal</thinking>Polish text"
    assert strip_model_thinking(text) == "Polish text"


# ---- repair validation (SPEC 8.3) ----------------------------------------
def test_good_repair_passes():
    check = check_repair_output(
        POLISH_CHUNK, POLISH_REPAIRED, context_before="przed", context_after="po"
    )
    assert check.ok, check.reason


def test_repair_with_diacritics_restored_plus_hyphen_join_passes():
    filler = " ".join(f"w{i}" for i in range(30))
    chunk = page(1, filler + "\npoprzed-\nnia czesc")
    out = page(1, filler + "\npoprzednia cześć")
    check = check_repair_output(chunk, out)
    assert check.ok, check.reason


def test_missing_marker_fails():
    check = check_repair_output(POLISH_CHUNK, POLISH_REPAIRED.replace("<<<PAGE 1>>>\n", ""))
    assert not check.ok
    assert "marker" in check.reason


def test_reordered_marker_fails():
    chunk = page(1, "a") + "\n\n" + page(2, "b")
    out = page(2, "b") + "\n\n" + page(1, "a")  # order swapped
    check = check_repair_output(chunk, out, context_before=CONTEXT_BEFORE, context_after=CONTEXT_AFTER)
    assert not check.ok
    assert "marker" in check.reason


def test_truncated_output_fails_word_count():
    check = check_repair_output(POLISH_CHUNK, POLISH_REPAIRED[:20])
    assert not check.ok
    assert "±35%" in check.reason


def test_chatty_output_fails():
    check = check_repair_output(POLISH_CHUNK, "Here is the repaired text:\n" + POLISH_REPAIRED)
    assert not check.ok
    assert "chat" in check.reason


def test_output_containing_context_fails():
    # keep the token count stable, but overwrite the chunk tail with a
    # verbatim slice of CONTEXT AFTER (simulates the model repeating context)
    marker_line, body = POLISH_CHUNK.split("\n", 1)
    body_words = body.split()
    swallowed = marker_line + "\n" + " ".join(
        body_words[:-16] + CONTEXT_AFTER.split()[:16]
    )
    check = check_repair_output(
        POLISH_CHUNK, swallowed, context_before=CONTEXT_BEFORE, context_after=CONTEXT_AFTER
    )
    assert not check.ok
    assert "context" in check.reason


def test_language_drift_fails():
    translated = page(
        1,
        "the that and of with was are the they this that the and was the of that they were that",
    )
    check = check_repair_output(POLISH_CHUNK, translated, context_before="w", context_after="w")
    assert not check.ok
    assert "drift" in check.reason


def test_ascii_folded_output_fails_diacritic_check():
    # damaged input still contains SOME diacritics; bad model folds them away
    chunk = page(1, "Życie na stół!")
    check = check_repair_output(chunk, page(1, "Zycie na stol!"), context_before="w", context_after="w")
    assert not check.ok
    assert "diacritics" in check.reason


def test_similarity_helpers():
    a = tokenize("raz dwa trzy", fold=True)
    assert sequence_similarity(a, a) == 1.0
    assert sequence_similarity(a, tokenize("raz dwa", fold=True)) > 0.5


# ---- the loose change envelope (user-tuned: 35% allowed changes) ----------
def test_repair_changes_inside_35_percent_envelope_accepted_with_warning():
    """~25% of words replaced: accepted, but flagged for diff review (<0.85)."""
    chunk = page(1, " ".join(f"w{i}" for i in range(100)))
    replaced = page(
        1, " ".join("naprawione" if i % 4 == 0 else f"w{i}" for i in range(100))
    )
    check = check_repair_output(chunk, replaced, context_before="a", context_after="b")
    assert check.ok, check.reason
    assert any("similarity below" in w for w in check.warnings)


def test_repair_changes_beyond_35_percent_envelope_rejected():
    """~50% of words replaced: clear hallucination territory -> fallback."""
    chunk = page(1, " ".join(f"w{i}" for i in range(100)))
    replaced = page(
        1, " ".join("naprawione" if i % 2 == 0 else f"w{i}" for i in range(100))
    )
    check = check_repair_output(chunk, replaced, context_before="a", context_after="b")
    assert not check.ok
    assert "too different" in check.reason


def test_format_changes_inside_35_percent_envelope_pass():
    """~34% of words rewritten but kept in order: formatter answer stands."""
    clean = page(1, " ".join(f"w{i}" for i in range(100)))
    changed = " ".join("inne" if i % 3 == 0 else f"w{i}" for i in range(100))
    check = check_format_output(clean, changed)  # formatter output: no markers
    assert check.ok, check.reason


def test_format_changes_beyond_35_percent_envelope_fall_back():
    clean = page(1, " ".join(f"w{i}" for i in range(100)))
    changed = " ".join("inne" if i % 2 == 0 else f"w{i}" for i in range(100))
    check = check_format_output(clean, changed)  # formatter output: no markers
    assert not check.ok
    # A 1:1 word replacement is seen by difflib as remove+add, so a
    # beyond-envelope replacement can trip either guard first; both reasons
    # mean "enough changed to fall back to the deterministic formatter".
    assert "word sequence" in check.reason or "added" in check.reason


# ---- format validation (SPEC 9, loosened to a 35% change envelope) --------
def test_format_small_additions_within_envelope_pass():
    # The loose envelope tolerates a few added words (e.g. a stray list
    # marker or one word the OCR mangled): 2 added on a 10-token chunk stays
    # inside the 35% slack, so the answer is accepted, not fallen back.
    chunk_input = page(1, "Rok dwadzieścia pierwszy.\nByła zima i padał śnieg na pola.")
    formatted = "Rok dwadzieścia pierwszy.\n\nByła zima i padał śnieg na pola, na drogi.\n"
    check = check_format_output(chunk_input, formatted)
    assert check.ok, check.reason


def test_format_added_words_beyond_envelope_fail():
    chunk_input = page(1, "Rok dwadzieścia pierwszy.\nByła zima i padał śnieg na pola.")
    formatted = (
        "# Rok dwadzieścia pierwszy\n\n"
        "Była zima i padał śnieg na pola, na drogi i w głębokim mrozie.\n"
    )
    check = check_format_output(chunk_input, formatted)
    assert not check.ok  # adds 6 invented words: beyond the 35% envelope
    assert "added" in check.reason


def test_format_added_heading_word_fails():
    chunk_input = page(1, "Była zima i padał śnieg na pola.")
    # A whole invented heading phrase: too many new words even in the envelope.
    formatted = "# Wielka Zima Bardzo Mrozna I Snezna\n\nByła zima i padał śnieg na pola.\n"
    check = check_format_output(chunk_input, formatted)
    assert not check.ok
    assert "added" in check.reason


def test_format_fine_passes():
    chunk_input = page(1, "Była zima i padał śnieg na pola.")
    formatted = "Była zima i padał śnieg na pola.\n"
    check = check_format_output(chunk_input, formatted)
    assert check.ok, check.reason


def test_format_removed_words_below_threshold_falls_back():
    many = " ".join(f"slowo{i}" for i in range(60))
    chunk_input = page(1, many)
    formatted = "slowo0 slowo59"  # dropped ~99% of words
    check = check_format_output(chunk_input, formatted)
    assert not check.ok
    assert "word sequence" in check.reason


def test_format_kept_marker_fails():
    chunk_input = page(1, "Test zdanie pierwsze.")
    formatted = "Test zdanie pierwsze.\n\n<<<PAGE 1>>>\n"
    check = check_format_output(chunk_input, formatted)
    assert not check.ok
    assert "marker" in check.reason


def test_running_header_removal_is_allowed():
    header = "MOJE PAMATNIKI"
    body = " ".join(f"w{i}" for i in range(120))
    chunk_input = page(1, header + "\n" + body)
    formatted = body  # header dropped deterministically
    check = check_format_output(chunk_input, formatted)
    assert check.ok, check.reason
    assert count_header_like_tokens(chunk_input) == len(tokenize(header))


def test_bare_page_number_removal_is_allowed():
    body = " ".join(f"w{i}" for i in range(120))
    chunk_input = page(1, "42\n" + body)
    check = check_format_output(chunk_input, body)
    assert check.ok, check.reason


def test_english_stopword_stats():
    # note: "to" is deliberately NOT a stopword here (it is a Polish word)
    count, share = english_stopword_stats(tokenize("the and of to the"))
    assert count == 4
    assert share == pytest.approx(4 / 5)
