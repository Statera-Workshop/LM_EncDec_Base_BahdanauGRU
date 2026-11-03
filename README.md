# LM_EncDec_Base_BahdanauGRU

Encoder-decoder translation model ported from legacy TF2 code C2 to PyTorch (cell=gru, attention=bahdanau).

## Status

- Verification level: **Benchmarked (LM-03, one run)**

## Changes from the legacy code

- PyTorch port; beam search reimplemented (TF Addons is no longer maintained)
- Vocabulary built from the training split only (legacy built it from all pairs)
- "what's" expanded to "what is" (legacy wrote "that is")

## LM-03 setting

English -> French, 5-11 words, 12,800 pairs (seed 1234), 80/20 split, embedding 128, hidden 1024, batch 256, 20 epochs, Adam 1e-3. Small data: compare structures, not absolute BLEU.

## Results

- Parameters: 29.5 M
- Test perplexity 30.8, next-word accuracy 0.580
- BLEU-4 greedy (whole test split): 22.3
- BLEU-4 beam width 4 on 500 pairs: 25.3 (greedy on the same pairs 23.9)
- Examples:
  - `this winter has been mild .` -> `cette piece est en train de cette .` (reference `cet hiver a ete doux .`)
  - `they will be here at three .` -> `ils sera ici a trois heures .` (reference `ils seront ici a trois heures .`)
  - `i once lived in rome .` -> `j'ai vecu de retour en arriere .` (reference `j'ai vecu un temps a rome .`)

## Run

```bash
pip install torch==2.5.1 einops numpy
# put fra-eng.zip from https://www.manythings.org/anki/ into data/
python run.py lm03 s2s_bahdanau_gru
```
