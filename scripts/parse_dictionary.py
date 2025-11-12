from __future__ import annotations

import re
import sys
import json
import typer
import yaml
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# You may switch to pdfminer.six if layout requires it.
from pypdf import PdfReader

from utils import (
    unhyphenate_lines,
    normalize_whitespace,
    compose_freq_band,
    compose_register_tags,
    sanitize_tsv_field,
    anki_header,
)

app = typer.Typer(
    help="Parse 'A Frequency Dictionary of Portuguese' PDF into an Anki-ready TSV.")

# --- Regex placeholders (fill these out during implementation) ---
# Notes:
# - Header line looks like: "{rank} {lemma} {pos} {gloss} [EP|BP]?"
# - Example lines: Portuguese sentence (line), then English translation (next line)
# - Frequency/register: "RANGE | RAWFREQ  +s -a" (numbers + optional +/- flags)
HEADER_RE = re.compile(
    r"^(?P<rank>\d+)\s+(?P<lemma>.+?)\s+(?P<pos>\w{1,4})\s+(?P<gloss>.+?)(?:\s+\[(?P<variety>EP|BP)\])?$")
FREQ_RE = re.compile(
    r"(?P<range>\d+)\s*\|\s*(?P<raw>\d+)(?P<flags>(?:\s+[+\-][sfna])*)")


def read_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n".join(pages)
    text = unhyphenate_lines(text)
    text = normalize_whitespace(text)
    return text


def load_config(cfg_path: Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compose_band_tags(rank: int, bands: List[Tuple[int, int]], families: List[str]) -> List[str]:
    tags = []
    band = compose_freq_band(rank, bands)
    if band != "NA":
        tags.append(f"freq::{band}")
        for fam in families or []:
            tags.append(f"{fam}::{band}")
    return tags


def parse_entries(raw_text: str) -> List[dict]:
    """
    Minimal stub parser to illustrate output shape.
    TODO: Implement full state machine for the dictionary layout.
    For now, this yields an empty list unless you add parsing logic.
    """
    entries: List[dict] = []
    # Implementation idea:
    # 1) iterate lines
    # 2) detect HEADER_RE -> start a new current entry
    # 3) next non-empty line -> Example_PT
    # 4) next line -> Example_EN
    # 5) look ahead for FREQ_RE -> capture range/raw/flags
    # 6) append entry, continue
    return entries


def build_dataframe(entries: List[dict], cfg: dict) -> pd.DataFrame:
    bands: List[Tuple[int, int]] = [(int(lo), int(hi))
                                    for lo, hi in cfg.get("freq_bands", [])]
    families: List[str] = cfg.get("freq_tag_families", [])
    pos_map: Dict[str, str] = cfg.get("pos_map", {})
    reg_map: Dict[str, str] = cfg.get("register_tags", {})
    variety_policy: str = cfg.get("variety_policy", "EP_only")

    rows = []
    for e in entries:
        rank = int(e.get("rank", 0))
        lemma = e.get("lemma", "").strip()
        pos_raw = e.get("pos", "").strip()
        gloss = e.get("gloss", "").strip()
        variety = e.get("variety", "").strip()  # "EP", "BP", or ""
        example_pt = e.get("example_pt", "").strip()
        example_en = e.get("example_en", "").strip()
        range_blocks = int(e.get("range_blocks", 0))
        raw_freq = int(e.get("raw_freq", 0))
        flags = e.get("flags", "").strip()

        # Apply variety policy
        if variety_policy == "EP_only" and variety == "BP":
            continue  # drop BP-dominant entries

        # Compute tags
        band_tags = compose_band_tags(rank, bands, families)
        reg_tags = compose_register_tags(flags, reg_map)
        variety_tag = [f"variety::{variety}"] if variety else []
        pos_tag = [f"pos::{pos_map.get(pos_raw, pos_raw)}"] if pos_raw else []

        tags = " ".join(band_tags + reg_tags + variety_tag + pos_tag).strip()

        rows.append({
            "Rank": rank,
            "Lemma_PT": lemma,
            "POS": pos_raw,
            "Gloss_EN": gloss,
            "Variety": variety,
            "Register_Flags": flags,
            "Range_Blocks": range_blocks,
            "Raw_Freq": raw_freq,
            "Example_PT": example_pt,
            "Example_EN": example_en,
            "Freq_Band": band_tags[0].split("::", 1)[1] if band_tags else "",
            "Thematic_Source": e.get("theme", ""),
            "Tags": tags,
        })

    df = pd.DataFrame(rows, columns=[
        "Rank", "Lemma_PT", "POS", "Gloss_EN", "Variety", "Register_Flags", "Range_Blocks", "Raw_Freq",
        "Example_PT", "Example_EN", "Freq_Band", "Thematic_Source", "Tags"
    ])
    return df


def write_tsv_with_anki_header(df: pd.DataFrame, cfg: dict, out_path: Path) -> None:
    sep = cfg.get("anki", {}).get("separator", "Tab")
    notetype = cfg.get("anki", {}).get(
        "notetype", "Portuguese (EP) – Frequency")
    deck = cfg.get("anki", {}).get("deck", "Portuguese::Frequency Master")
    columns = list(df.columns)
    header = anki_header(sep, notetype, deck, columns)

    cleaned = df.copy()
    for col in cleaned.columns:
        cleaned[col] = cleaned[col].astype(str).map(sanitize_tsv_field)

    tsv = cleaned.to_csv(sep="\t", index=False, line_terminator="\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(header)
        f.write(tsv)


def write_qc_sample(df: pd.DataFrame, qc_path: Path, n: int = 50) -> None:
    sample = df.sample(n=min(n, len(df)), random_state=42) if len(
        df) else df.head(0)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(qc_path, index=False, encoding="utf-8")


@app.command()
def parse(
    pdf: Path = typer.Option(..., "--pdf",
                             help="Path to the frequency dictionary PDF"),
    out: Path = typer.Option(..., "--out",
                             help="Path to write the Anki-ready TSV"),
    qc: Path = typer.Option("out/qc_sample.csv", "--qc",
                            help="Optional QC sample CSV"),
    cfg: Path = typer.Option("config.yaml", "--cfg",
                             help="Path to config.yaml"),
):
    """
    Parse the PDF and export an Anki-ready TSV with headers.
    """
    typer.echo(f"[info] Loading config from {cfg}")
    cfg_data = load_config(cfg)

    typer.echo(f"[info] Reading PDF text from {pdf}")
    try:
        raw_text = read_pdf_text(pdf)
    except Exception as e:
        typer.secho(
            f"[warn] PDF read failed ({e}). Continuing with empty text for a dry run.", fg=typer.colors.YELLOW)
        raw_text = ""

    typer.echo("[info] Parsing entries (stub implementation)")
    entries = parse_entries(raw_text)

    typer.echo(f"[info] Parsed {len(entries)} entries; building DataFrame")
    df = build_dataframe(entries, cfg_data)

    typer.echo(f"[info] Writing TSV to {out}")
    write_tsv_with_anki_header(df, cfg_data, out)

    typer.echo(f"[info] Writing QC sample to {qc}")
    write_qc_sample(df, Path(qc))

    typer.secho(
        "[done] Finished. Fill in parse_entries() next to extract real rows.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
