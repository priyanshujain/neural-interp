from pathlib import Path

import numpy as np

from neural_interp.lib import gemma

N_LINES = 400_000
BATCH = 10_000


def run(out):
    out = Path(out)
    tok = gemma.tokenizer()
    lines = [t for t in gemma.wikitext() if t.strip()][:N_LINES]
    counts = np.zeros(len(tok), dtype=np.int64)
    for start in range(0, len(lines), BATCH):
        ids = tok(lines[start:start + BATCH], add_special_tokens=False)["input_ids"]
        counts += np.bincount(np.concatenate([np.asarray(i, dtype=np.int64) for i in ids]), minlength=len(tok))
        print("lines", start + BATCH, "tokens", int(counts.sum()), flush=True)
    kept = np.load(out / "token_ids.npy")
    np.save(out / "freq.npy", counts[kept])
    print("kept tokens seen at least once", int((counts[kept] > 0).sum()), "of", len(kept))
