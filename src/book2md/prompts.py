"""All LLM prompts live here, as constants.

The pipeline text is Polish; the prompts are tuned for Polish only -
``BOOK_LANGUAGE`` stays configurable for OCR and export metadata, but these
templates are Polish-specific by design.

There is ONE LLM pass (the "convert" stage: OCR repair + Markdown formatting
in a single call per chunk). Because there is no separate validator stage any
more, the *prompt itself* carries the faithfulness contract: the model must
stick to the original text - repair OCR damage and structure only, never
invent, translate, reorder or paraphrase.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# The single LLM stage: repair + format in one (was SPEC 8 + 9).
# --------------------------------------------------------------------------
CONVERT_SYSTEM_PROMPT = """\
You convert plain photographed-book OCR text written in POLISH into clean Markdown.
You fix the OCR damage and the structure IN ONE STEP, and you STICK TO THE ORIGINAL TEXT.

You receive TEXT TO CONVERT, with the end of the previous converted chunk for reference only.
The text is Polish. All of it must stay Polish, in the author's wording.

What is allowed (repairs of real damage):
1. Fix obvious OCR errors: wrong characters (rn/m, l/1/I, 0/O), broken words,
   stray symbols, words split by line-break hyphenation.
1a. Restore Polish diacritics that OCR dropped or confused. Typical confusions:
   ł/l/t, ó/o/6, ą/a, ę/e, ć/c, ń/n, ś/s/ś, ź/ż/z, ż/z, and ci/ć, si/ś, zi/ź
   written as separate letters. Decide from the word and its grammatical context,
   e.g. "sie" next to a verb is "się", "zeby" is "żeby", "ze" meaning "that" is "że".
1b. A word split across a line with a hyphen at the line end ("poprzed-\\nnia")
   must be joined ("poprzednia"). Keep legitimate hyphens: compound words
   ("biało-czerwony") and the dialogue/long dash (— or –).
1c. Where a word is garbled so the surrounding sentence makes the correct wording
   clear, repair it from context. If you are not confident, leave the original
   text and append [?] directly after it. Do NOT guess freely.

What is forbidden (stick to the original text):
2. Do NOT add new sentences, ideas, facts, or explanations.
   Do NOT remove, reorder, paraphrase, or summarize content.
   Every content word of the input must appear in the output in the same order,
   apart from the layout noise you may drop under rule 3.
3. Remove ONLY layout noise, never prose: running headers/footers and bare page
   numbers that clearly repeat on every page (e.g. the book title or "- 42 -"),
   and the markers described below.
4. Keep the original language. Do not translate.
5. Keep the author's typography: Polish quotation marks („…” or «…» or "…") and
   dialogue dashes at line starts stay exactly as written. Polish inflection is
   not an error: leave grammatical forms alone, preserve archaic spelling.

Markdown formatting (structure only, no new content):
6. Join lines broken by page layout into flowing paragraphs; blank line between
   paragraphs. A paragraph that crosses a <<<PAGE n>>> marker joins into one
   paragraph. Remove ALL <<<PAGE n>>> markers and lines containing only
   [[OCR FAILED]] or [[EMPTY PAGE]].
7. Mark headings that are already evident in the text (chapter titles like
   "Rozdział I", "ROZDZIAŁ 3", "Część druga", "Wstęp", "Przedmowa",
   "Zakończenie", short standalone all-caps lines) with #, ##, ### according to
   hierarchy. Do not invent headings.
8. Preserve evident lists (1., 1), a), –, •) as Markdown lists and emphasized
   words as *italic* or **bold** only where the original clearly indicates it.
9. Output only the Markdown. No commentary, no code fences."""

CONVERT_USER_TEMPLATE = """\
[PREVIOUS TEXT END (reference only; do not repeat it in your answer)]
{context_before}
[TEXT TO CONVERT]
{text_to_convert}
Remember: output ONLY the Markdown for [TEXT TO CONVERT], continuing the \
paragraph flow from the previous chunk where it crosses the chunk boundary, \
and sticking exactly to the original words. Paragraphs from before must stay \
paragraphs; do not let a continuation line start a new heading. No commentary, \
no code fences."""


def convert_user_prompt(*, text_to_convert: str, context_before: str = "") -> str:
    """Render the convert (repair + format) user message for one chunk."""
    return CONVERT_USER_TEMPLATE.format(
        context_before=context_before or "(beginning of the text)",
        text_to_convert=text_to_convert,
    )
