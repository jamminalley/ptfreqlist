#!/usr/bin/env python
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import pdfplumber
import pandas as pd
import typer
import yaml

app = typer.Typer()

DEFAULT_CFG = Path("config.yaml")

# POS codes we expect in the dictionary
POS_CODES = (
    # nouns
    "nm", "nf", "n", "na", "nmf",
    # adjectives
    "adj", "aj",
    # verbs
    "v",
    # adverbs
    "adv", "av",
    # prepositions
    "prep", "prp", "aprp",
    # conjunctions
    "conj", "cj", "ecj",
    # pronouns
    "pron", "pn",
    # numerals / determiners / articles (just in case)
    "num", "art", "det",
)

# Match an entry start anywhere in the text: rank + lemma + POS
ENTRY_START_RE = re.compile(
    r"(?P<rank>[1-9]\d{0,3})\s+"
    r"(?P<lemma>[^\s]+)\s+"
    r"(?P<pos>" + "|".join(POS_CODES) + r")\s+",
    re.UNICODE,
)


def read_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_pdf_text(pdf_path: Path) -> str:
    """Read all text from the trimmed PDF."""
    text_parts: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_parts.append(t)
    return "\n".join(text_parts)


def normalise_spaces(text: str) -> str:
    return " ".join(text.split())


def strip_book_cruft(text: str) -> str:
    """
    Remove obvious book-level cruft that can leak into entries.

    - Book title: 'A Frequency Dictionary of Portuguese'
    - Section marker: 'Frequency Index <number>'
    - Thematic list block like '1.Animals ... Frequency Index ...'
      which otherwise gets glued onto entry 10 'a'.
    """
    # remove book title if it shows up inside an entry
    text = text.replace("A Frequency Dictionary of Portuguese", "")

    # remove specific Animals thematic block if present
    # (everything from '1.Animals' up to 'Frequency Index ...' or end of string)
    text = re.sub(
        r"\b1\.Animals\b.*?(?=Frequency Index\b|$)",
        "",
        text,
        flags=re.DOTALL,
    )

    # remove 'Frequency Index <number>' fragments that might remain
    text = re.sub(r"Frequency Index\s+\d+", "", text)

    # clean up any doubled spaces left behind
    return normalise_spaces(text)


def split_entries(raw_text: str) -> List[str]:
    """
    Split the full text into chunks, one per dictionary entry.

    We look for every occurrence of 'rank lemma POS' and treat that
    as the start of an entry, regardless of where it appears in a line.
    """
    matches = list(ENTRY_START_RE.finditer(raw_text))
    chunks: List[str] = []

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        chunk = raw_text[start:end]
        chunks.append(chunk)

    return chunks


def parse_rest(rest: str):
    """
    Parse the 'rest' of an entry after rank/lemma/pos.

    We try to recover:
      - gloss_en
      - block_count (Range_Blocks)
      - raw_freq (Raw_Freq)
      - example_pt
      - example_en
      - freq_band
      - thematic_source (optional; we leave empty for now)
    """
    rest = normalise_spaces(rest)
    rest = strip_book_cruft(rest)

    # Frequency band is usually the last "dddd-dddd" chunk
    band_match = re.search(r"(\d{4}-\d{4})", rest)
    freq_band = ""
    if band_match:
        freq_band = band_match.group(1)
        main = rest[: band_match.start()].strip()
    else:
        main = rest.strip()

    # Split out examples using bullet and en dash
    example_pt = ""
    example_en = ""

    bullet_idx = main.find("•")
    if bullet_idx != -1:
        head = main[:bullet_idx].strip()
        ex_part = main[bullet_idx + 1:].strip()
        dash_idx = ex_part.find("–")
        if dash_idx != -1:
            example_pt = ex_part[:dash_idx].strip()
            example_en = ex_part[dash_idx + 1:].strip()
        else:
            example_pt = ex_part
    else:
        head = main

    # At the very end of 'head' we expect "block_count raw_freq"
    block_count = None
    raw_freq = None
    m = re.search(r"(\d+)\s+(\d+)\s*$", head)
    if m:
        block_count = int(m.group(1))
        raw_freq = int(m.group(2))
        head_main = head[: m.start()].strip()
    else:
        head_main = head

    gloss_en = head_main
    thematic_source = ""  # we’re not reliably separating this, so leave blank

    return gloss_en, block_count, raw_freq, example_pt, example_en, freq_band, thematic_source


def freq_band_from_rank(rank: int, bands: List[List[int]]) -> str:
    """Fallback: compute frequency band from the rank and config bands."""
    for start, end in bands:
        if start <= rank <= end:
            return f"{start:04d}-{end:04d}"
    return ""


def build_tags(pos_code: str, long_pos: str, band_str: str, cfg: Dict[str, Any]) -> str:
    tags: List[str] = []

    if band_str:
        tags.append(f"freq::{band_str}")
        for fam in cfg.get("freq_tag_families", []):
            tags.append(f"{fam}::{band_str}")

    tags.append(f"pos::{long_pos}")
    return " ".join(tags)


def parse_entry(chunk: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a single entry chunk into a row dict.
    """
    m = ENTRY_START_RE.match(chunk)
    if not m:
        raise ValueError(
            "No entry header (rank/lemma/POS) found at start of chunk")

    rank = int(m.group("rank"))
    lemma = m.group("lemma")
    pos_code = m.group("pos")

    rest = chunk[m.end():].strip()

    (
        gloss_en,
        block_count,
        raw_freq,
        example_pt,
        example_en,
        band_from_text,
        thematic_source,
    ) = parse_rest(rest)

    bands = cfg.get("freq_bands", [])
    freq_band = band_from_text or freq_band_from_rank(rank, bands)

    variety_policy = cfg.get("variety_policy", "")
    variety = "EP" if variety_policy == "EP_only" else ""

    pos_map = cfg.get("pos_map", {})
    long_pos = pos_map.get(pos_code, pos_code)

    tags = build_tags(pos_code, long_pos, freq_band, cfg)

    row: Dict[str, Any] = {
        "Rank": rank,
        "Lemma_PT": lemma,
        "POS": pos_code,
        "Gloss_EN": gloss_en,
        "Variety": variety,
        "Register_Flags": "",
        "Range_Blocks": block_count,
        "Raw_Freq": raw_freq,
        "Example_PT": example_pt,
        "Example_EN": example_en,
        "Freq_Band": freq_band,
        "Thematic_Source": thematic_source,
        "Tags": tags,
    }

    return row


def write_tsv_with_anki_header(df: pd.DataFrame, cfg: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    anki_cfg = cfg.get("anki", {})
    deck = anki_cfg.get("deck", "Portuguese::Frequency Master")
    notetype = anki_cfg.get("notetype", "Portuguese (EP) – Frequency")
    separator_name = anki_cfg.get("separator", "Tab")

    # For now, we always write tab-separated, but keep the separator name in the header.
    sep = "\t"

    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# deck: {deck}\n")
        f.write(f"# notetype: {notetype}\n")
        f.write(f"# separator: {separator_name}\n")
        df.to_csv(f, sep=sep, index=False)


def write_qc_sample(df: pd.DataFrame, qc_path: Path, n: int = 50) -> None:
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        sample = df.copy()
    else:
        sample = df.sample(n=min(n, len(df)),
                           random_state=0).sort_values("Rank")

    # utf-8-sig so Excel shows the diacritics correctly
    sample.to_csv(qc_path, index=False, encoding="utf-8-sig")


@app.command()
def parse(
    pdf: Path = typer.Option(
        ...,
        "--pdf",
        help="Path to trimmed A_Frequency_Dictionary_Of_Portuguese PDF",
        exists=True,
        readable=True,
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Output TSV file for Anki deck",
    ),
    qc: Path = typer.Option(
        ...,
        "--qc",
        help="Output CSV file with a QC sample",
    ),
    cfg: Path = typer.Option(
        DEFAULT_CFG,
        "--cfg",
        help="YAML config file",
    ),
    demo: bool = typer.Option(
        False,
        "--demo",
        help="Parse only the first ~200 entries for quick testing",
    ),
) -> None:
    typer.echo(f"[info] Loading config from {cfg}")
    cfg_data = read_config(cfg)

    typer.echo(f"[info] Reading PDF text from {pdf}")
    raw_text = read_pdf_text(pdf)

    typer.echo("[info] Splitting into candidate entries…")
    chunks = split_entries(raw_text)
    if demo:
        chunks = chunks[:200]
    typer.echo(f"[info] Found {len(chunks)} candidate entry chunks")

    rows: List[Dict[str, Any]] = []
    for chunk in chunks:
        try:
            row = parse_entry(chunk, cfg_data)
            rows.append(row)
        except Exception as e:
            preview = normalise_spaces(chunk)[:120]
            typer.echo(
                f"[warn] Could not parse entry starting '{preview}': {e}")

    df = pd.DataFrame(rows)
    df = df.sort_values("Rank").reset_index(drop=True)

    typer.echo(f"[info] Parsed {len(df)} entries; writing TSV to {out}")
    write_tsv_with_anki_header(df, cfg_data, out)

    typer.echo(f"[info] Writing QC sample to {qc}")
    write_qc_sample(df, qc)

    typer.secho("[done] Finished.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
