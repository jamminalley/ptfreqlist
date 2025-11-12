# pt-frequency-anki

Build an Anki-ready TSV from *A Frequency Dictionary of Portuguese* (Davies).  
One **note** per lemma with fields for rank, lemma, POS, gloss, variety (EP/BP), register flags, range, raw freq, example (PT/EN), optional themes, and computed frequency bands.  
Anki templates can generate both **receptive (PT→EN)** and **productive (EN→PT)** cards; you can enable/disable either set later.

## Quick start (Windows 11)
1. Open VS Code in this folder.
2. Create a virtual environment and install requirements:
   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt

