"""Anti-hallucination checks (SPEC 8.3, 9) - the most important quality rule.

Every LLM chunk answer is checked; on any failure the chunk **falls back to
its unrepaired/unformatted counterpart** and a warning is logged. The LLM
stages may only correct, fill obvious gaps (marked with ``[?]``), and format.

Polish-specific rules implemented here:

* NFC normalization before comparison; ``casefold()`` only for folding case.
* Diacritics must not disappear: the share of ``ąćęłńóśźż...`` among letters
  may drop by at most 20% versus the input.
* Language drift: many English stopwords entering a Polish text means the
  model translated -> reject.
* Comparisons never use diacritic stripping on the *word-sequence* level for
  the formatter; diacritic folding happens ONLY inside the repair-stage
  similarity/context-overlap helpers, where the repair step legitimately
  *changes* spelling by restoring diacritics (and a dedicated share-check
  guarantees it did not silently *fold* them instead).

How much may change at all is governed by :data:`ALLOWED_CHANGE_RATIO`
(user request after noisy real-world runs: up to 35% of words per chunk may
differ before a stage falls back - see the constant for details).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .chunking import PAGE_MARKER_RE, word_count

#: Polish diacritic alphabet whose preservation we police (SPEC 8.3).
POLISH_DIACRITICS = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"
_DIACRITIC_SET = set(POLISH_DIACRITICS)

#: Relative floor for the diacritic share (may not drop below 80% of input).
DIACRITIC_KEEP_RATIO = 0.8

#: User-tuned change envelope (deliberate loosening of SPEC 8.3/9 thresholds).
#:
#: Photographed-book OCR is noisy, so an over-strict validator kept falling
#: back to the (still damaged) raw text. From now on a chunk answer may differ
#: from its input by **up to 35% of its words** (replaced, removed or added)
#: before the stage declares the chunk broken and uses the fallback. This ONE
#: constant drives all four meaningful limits, so they stay consistent:
#:
#: * repair similarity floor            : 1 - 0.35 = 0.65
#: * formatter word-sequence threshold  : 1 - 0.35 = 0.65
#: * repair word-count band             : +/- 35%
#: * formatter added-word slack         : 35% of the input tokens
#:
#: Safety checks that are NOT affected: page markers, chatter, swallowed
#: context blocks, language drift and the diacritic-collapse guard (they are
#: about wrong behaviour, not about how much was changed).
ALLOWED_CHANGE_RATIO = 0.35

#: Repair similarity thresholds (SPEC 8.3 wanted "< 0.85 -> flag it"; the
#: hard floor below which the chunk falls back is now LOOSE, see
#: :data:`ALLOWED_CHANGE_RATIO`).
SIMILARITY_REJECT = 1.0 - ALLOWED_CHANGE_RATIO   # 0.65: >35% changed -> fallback
SIMILARITY_WARN = 0.85   # below this a warning is logged, chunk still accepted

#: Formatter word-sequence threshold (SPEC 9 had 0.97; loosened via
#: :data:`ALLOWED_CHANGE_RATIO` for noisy OCR): below 0.65 -> deterministic fallback.
FORMAT_SIMILARITY_THRESHOLD = 1.0 - ALLOWED_CHANGE_RATIO

#: English stopwords used ONLY for the language-drift check; deliberately
#: excludes words that double as common Polish words (e.g. "to", "in", "it",
#: "not", "for" - their Polish lookalikes are separate letter-strings, but
#: being conservative avoids false positives on mixed-language books).
ENGLISH_STOPWORDS = frozenset(
    """the and of is that this with as was are be have has had would could should
    about there their they were which while from""".split()
)

#: Leading chatter the model was forbidden to produce (SPEC 8.3).
_CHATTER_RE = re.compile(
    r"^\s*(?:here (?:is|'s|'re)|here you go|sure|okay|certainly|of course|"
    r"as (?:requested|promised|specified|follows)|below (?:is|are|the)|"
    r"the (?:repaired|formatted|output)|i (?:have|had|'ve|will|am|would)|"
    r"how about|let's)\b",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

#: Sliding-window size for "did the output swallow a context block?" (words).
CONTEXT_NGRAM = 16

#: Placeholders merged into the raw file for failed/empty OCR pages.
OCR_PLACEHOLDERS = ("[[OCR FAILED]]", "[[EMPTY PAGE]]")


def normalize_nfc(text: str) -> str:
    """Unicode NFC normalization - applied to all pipeline text (SPEC 6/8.3)."""
    return unicodedata.normalize("NFC", text)


def fold_diacritics(text: str) -> str:
    """Map Polish diacritics to base letters ('się' -> 'sie') for *similarity*.

    Only used by the repair stage: LLM repairs legitimately re-spell words
    (restoring diacritics), so token equality would misjudge a good repair.
    The companion share-check (:func:`diacritic_share`) ensures the model did
    not simply ASCII-fold everything.
    """
    table = str.maketrans(POLISH_DIACRITICS, "acelnoszzACELNOSZZ")
    return text.translate(table)


def tokenize(
    text: str, *, fold: bool = False, drop_markers: bool = False
) -> list[str]:
    """Word tokens: NFC, ``casefold()``, punctuation/markup stripped.

    Diacritics are deliberately KEPT (SPEC 8.3: "Do not use diacritic-stripping
    when comparing words anywhere in the tokenizer"). ``fold=True`` is the
    repair-stage-only exception documented in :func:`fold_diacritics`.
    ``drop_markers=True`` also removes ``<<<PAGE n>>>`` (strip before format
    checks: the formatter is *required* to remove them).
    """
    text = normalize_nfc(text)
    if drop_markers:
        text = PAGE_MARKER_RE.sub("", text)
    if fold:
        text = fold_diacritics(text)
    return [tok.casefold() for tok in _WORD_RE.findall(text)]


def diacritic_share(text: str) -> float:
    """Share of Polish diacritic characters among letters.

    ``0.0`` for ASCII-only text; used for the "model simplified to ASCII"
    rejection (SPEC 8.3).
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    hits = sum(1 for ch in letters if ch in _DIACRITIC_SET)
    return hits / len(letters)


def hard_check_note(text: str) -> str | None:
    """Shared pre-checks: empty output, leading chatter, accidental tags.

    Returns the failure reason or ``None`` when the text looks sane so far.
    """
    candidate = text.strip()
    if not candidate:
        return "empty output"
    probe = candidate[:140]
    if _CHATTER_RE.match(probe):
        return f"chat-like preamble instead of pure text: {probe[:60]!r}"
    # XML-ish wrappers (<output>...) are forbidden; page markers are not tags.
    if candidate.startswith("<") and not candidate.startswith("<<<PAGE"):
        return "output starts with markup, not text"
    return None


def strip_code_fences(text: str) -> str:
    """Remove one leading/trailing ``` fence pair - salvage, not validation.

    Models stubbornly wrap whole documents in fences even when forbidden;
    the *content* is then perfectly valid, so strip and validate normally.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            lines = lines[1:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def strip_model_thinking(text: str) -> str:
    """Remove ``indsight``/``thinking`` blocks some providers echo back.

    Salvage step: only whole wrapping blocks are removed, real text stays.
    """
    return re.sub(
        r"<thinking>.*?</thinking>|<thought>.*?</thought>|<reflection>.*?</reflection>",
        "",
        text,
        flags=re.DOTALL,
    )


def _context_contains(output_tokens: list[str], context_tokens: list[str]) -> bool:
    """Does any CONTEXT_NGRAM-gram of the context appear in the output?

    Used for "the output does not contain the context blocks" (SPEC 8.3).
    A positional index keeps this fast even for 2,000-word chunks.
    """
    if len(context_tokens) < CONTEXT_NGRAM:
        return False
    positions: dict[str, list[int]] = {}
    for i, token in enumerate(output_tokens):
        positions.setdefault(token, []).append(i)
    for start in range(len(context_tokens) - CONTEXT_NGRAM + 1):
        first = context_tokens[start]
        for idx in positions.get(first, ()):
            if output_tokens[idx : idx + CONTEXT_NGRAM] == context_tokens[
                start : start + CONTEXT_NGRAM
            ]:
                return True
    return False


def english_stopword_stats(tokens: list[str]) -> tuple[int, float]:
    """``(count, share)`` of English stopwords among tokens (drift check)."""
    count = sum(1 for tok in tokens if tok in ENGLISH_STOPWORDS)
    share = count / len(tokens) if tokens else 0.0
    return count, share


def check_repair_output(
    chunk_input: str,
    output: str,
    *,
    context_before: str = "",
    context_after: str = "",
) -> RepairCheck:
    """Full validation of one repair answer (SPEC 8.3).

    Order matters: cheap checks first, similarity last. ``output`` must have
    gone through :func:`strip_code_fences` / :func:`strip_model_thinking` first
    (the caller does that).
    """
    chatter = hard_check_note(output)
    if chatter:
        return RepairCheck(ok=False, reason=chatter)

    in_markers = PAGE_MARKER_RE.findall(chunk_input)
    out_markers = PAGE_MARKER_RE.findall(output)
    if in_markers != out_markers:
        return RepairCheck(
            ok=False,
            reason=f"page markers lost, reordered or duplicated "
            f"(in={len(in_markers)}, out={len(out_markers)})",
        )

    in_words = word_count(normalize_nfc(chunk_input))
    out_words = word_count(normalize_nfc(output))
    if in_words and not (
        (1.0 - ALLOWED_CHANGE_RATIO) * in_words
        <= out_words
        <= (1.0 + ALLOWED_CHANGE_RATIO) * in_words
    ):
        return RepairCheck(
            ok=False,
            reason=f"word count changed beyond ±{ALLOWED_CHANGE_RATIO:.0%} "
            f"({in_words} -> {out_words})",
        )

    in_tokens_fold = tokenize(chunk_input, fold=True, drop_markers=True)
    out_tokens_fold = tokenize(output, fold=True, drop_markers=True)
    context_tokens = tokenize(
        (context_before + " " + context_after), fold=True, drop_markers=True
    )
    if context_tokens and _context_contains(out_tokens_fold, context_tokens):
        return RepairCheck(ok=False, reason="output repeats a context block")

    # Language drift: English stopwords appearing where Polish text has none.
    in_count, in_share = english_stopword_stats(in_tokens_fold)
    out_count, out_share = english_stopword_stats(out_tokens_fold)
    if (
        out_count >= 10
        and out_share >= 0.02
        and out_count > 5 * max(in_count, 1)
    ):
        return RepairCheck(
            ok=False,
            reason=f"language drift: {out_count} English stopwords vs {in_count} "
            f"in input (model likely translated)",
        )

    # Diacritic preservation: NFC-normalized share may not drop more than 20%.
    in_share_dia = diacritic_share(normalize_nfc(strip_page_marker_lines(chunk_input)))
    out_share_dia = diacritic_share(normalize_nfc(strip_page_marker_lines(output)))
    if in_share_dia > 0.0:
        if out_share_dia < DIACRITIC_KEEP_RATIO * in_share_dia:
            return RepairCheck(
                ok=False,
                reason="diacritics dropped: share "
                f"{in_share_dia:.3f} -> {out_share_dia:.3f} (model ASCII-folded?)",
            )

    # Normalized similarity (difflib on diacritic-folded words).
    similarity = sequence_similarity(in_tokens_fold, out_tokens_fold)
    warnings: list[str] = []
    if similarity < SIMILARITY_REJECT:
        return RepairCheck(
            ok=False,
            reason=f"output too different from input (similarity {similarity:.3f})",
        )
    if similarity < SIMILARITY_WARN:
        warnings.append(f"similarity below {SIMILARITY_WARN}: {similarity:.3f}, review the diff")
    return RepairCheck(ok=True, similarity=similarity, warnings=warnings)


def strip_page_marker_lines(text: str) -> str:
    """Markers only (no surrounding words) - for diacritic-share comparisons."""
    return PAGE_MARKER_RE.sub("", text)


@dataclass
class RepairCheck:
    """Result of :func:`check_repair_output`."""

    ok: bool
    reason: str | None = None
    similarity: float | None = None
    warnings: list[str] = field(default_factory=list)


def sequence_similarity(a: list[str], b: list[str]) -> float:
    """difflib similarity over token sequences (SPEC 8.3 optional check)."""
    if not a and not b:
        return 1.0
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    return matcher.ratio()


# --------------------------------------------------------------------------
# Formatter validation (SPEC 9): word-sequence check with a loose fallback
# floor (35% of words may differ -> deterministic fallback below that).
# --------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s")
_BARE_PAGE_NUMBER_RE = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")

#: Short all-caps-ish standalone lines are probable running headers, and the
#: formatter is allowed (SPEC 9, rule 6) to remove them.
_HEADER_LINE_MAX_WORDS = 6


def count_header_like_tokens(text: str) -> int:
    """Estimate how many input tokens the formatter may legitimately delete.

    Counts tokens of: bare page-number lines (``- 42 -``, ``42``) and short
    all-caps lines (running headers like ``MOJE PANSTWO``). This slack keeps
    the 97% threshold honest for chunks that contain many headers/footers
    instead of punishing the model for removing them (SPEC 9, rule 6).
    """
    removed = 0
    for line in strip_page_marker_lines(text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _BARE_PAGE_NUMBER_RE.match(stripped):
            removed += len(tokenize(stripped))
            continue
        words = _WORD_RE.findall(stripped)
        if (
            len(words) <= _HEADER_LINE_MAX_WORDS
            and words
            and stripped == stripped.upper()  # no lowercase letters anywhere
            and any(ch.isalpha() for ch in stripped)
        ):
            # all-caps standalone line (running header candidate)
            removed += len(words)
    return removed


def check_format_output(chunk_input: str, output: str) -> FormatCheck:
    """Word-sequence check for the formatter (SPEC 9).

    Tokenize input and output (NFC, casefolded, punctuation and Markdown
    symbols stripped, diacritics kept). Output tokens must match input tokens,
    except for removed running headers/page numbers and markers (markers are
    stripped from the input side; the formatter must remove them). Chunks
    below the threshold fall back to a deterministic formatter.
    """
    chatter = hard_check_note(output)
    if chatter:
        return FormatCheck(ok=False, reason=chatter)
    if "<<<PAGE" in output:
        return FormatCheck(ok=False, reason="formatter left a page marker in")

    in_tokens = tokenize(chunk_input, drop_markers=True)
    out_tokens = tokenize(output, drop_markers=True)

    matcher = SequenceMatcher(None, in_tokens, out_tokens, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    removed = len(in_tokens) - matched
    added = len(out_tokens) - matched

    allowed_removed = count_header_like_tokens(chunk_input)
    # Markdown syntax (---, **, #) is not tokenized. Inside the loose control
    # envelope the model may likewise add up to ALLOWED_CHANGE_RATIO of the
    # input tokens (fixing words the OCR mangled entirely); attempts to add
    # more than that are rejected as new content (SPEC 9 rule 1).
    added_slack = int(ALLOWED_CHANGE_RATIO * max(len(in_tokens), 1))

    similarity = matched / len(in_tokens) if in_tokens else 1.0

    if added > added_slack:
        return FormatCheck(
            ok=False,
            similarity=similarity,
            reason=f"formatter added {added} words beyond the input "
            f"(slack {added_slack}); new content is forbidden",
        )
    # Word-sequence envelope: all header-like tokens plus a flexible budget
    # (ALLOWED_CHANGE_RATIO for replacements/typos the OCR left behind) may
    # differ; anything below the resulting match floor falls back.
    flexible_budget = int(ALLOWED_CHANGE_RATIO * max(len(in_tokens), 1))
    required = max(
        0,
        len(in_tokens) - allowed_removed - flexible_budget,
    )
    if matched < required:
        return FormatCheck(
            ok=False,
            similarity=similarity,
            reason=(
                f"word sequence changed: kept {matched}/{len(in_tokens)} "
                f"(header-like {allowed_removed} + flexible {flexible_budget}, "
                f"similarity {similarity:.4f} < {FORMAT_SIMILARITY_THRESHOLD})"
            ),
        )
    warnings: list[str] = []
    if removed > 0:
        warnings.append(f"formatter removed {removed} words (headers/page numbers?)")
    return FormatCheck(
        ok=True,
        similarity=similarity,
        warnings=warnings,
        removed_tokens=removed,
    )


@dataclass
class FormatCheck:
    """Result of :func:`check_format_output`."""

    ok: bool
    reason: str | None = None
    similarity: float | None = None
    warnings: list[str] = field(default_factory=list)
    removed_tokens: int = 0
