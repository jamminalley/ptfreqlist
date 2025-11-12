from __future__ import annotations

import re
from typing import Dict, List, Tuple


def unhyphenate_lines(text: str) -> str:
    """
    Join hyphenated line breaks common in PDFs.
    Example: "con-\ntinuar" -> "continuar".
    """
    # remove hyphen at EOL followed by newline and a lowercase letter
    return re.sub(r"-\n(?=[a-zà-ü])", "", text, flags=re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00A0", " ")  # non-breaking space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def compose_freq_band(rank: int, bands: List[Tuple[int, int]]) -> str:
    for lo, hi in bands:
        if lo <= rank <= hi:
            return f"{lo:04d}-{hi:04d}"
    return "NA"


def compose_register_tags(flag_blob: str, reg_map: Dict[str, str]) -> List[str]:
    """
    From flag blob like '+s -a' produce tags:
    ['register::plus_spoken', 'register::minus_academic']
    """
    tags = []
    for code, human in reg_map.items():
        # look for +code or -code as separate tokens
        if re.search(rf"(?:^|\s)\+{re.escape(code)}(?:\s|$)", flag_blob):
            tags.append(f"register::plus_{human}")
        if re.search(rf"(?:^|\s)-{re.escape(code)}(?:\s|$)", flag_blob):
            tags.append(f"register::minus_{human}")
    return tags


def sanitize_tsv_field(s: str) -> str:
    """
    Keep it simple for TSV: replace hard newlines with <br> so Anki can render if Allow HTML is enabled.
    """
    return s.replace("\t", " ").replace("\r", "").replace("\n", "<br>").strip()


def anki_header(separator: str, notetype: str, deck: str, columns: List[str]) -> str:
    sep_line = f"#separator: {separator}"
    note_line = f"#notetype: {notetype}"
    deck_line = f"#deck: {deck}"
    cols = "\t".join(columns) if separator.lower(
    ) == "tab" else ",".join(columns)
    cols_line = "#columns: " + cols
    return "\n".join([sep_line, note_line, deck_line, cols_line]) + "\n"
