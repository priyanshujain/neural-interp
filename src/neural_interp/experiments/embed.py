import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch

from neural_interp.lib import gemma


def is_byte_piece(text):
    return bool(re.fullmatch(r"<0x[0-9A-F]{2}>", text))


def script_of(text):
    if is_byte_piece(text):
        return "BYTE"
    stripped = text.strip()
    if not stripped:
        return "SPACE"
    for ch in stripped:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            return name.split()[0] if name else "OTHER"
    return "DIGIT" if any(ch.isdigit() for ch in stripped) else "SYMBOL"


def undertrained(W, is_unused, is_special):
    centred = torch.from_numpy(W - W.mean(0))
    first_pc = torch.pca_lowrank(centred, q=1, center=False)[2][:, 0]
    without_pc = centred - (centred @ first_pc)[:, None] * first_pc[None, :]
    reference = without_pc[torch.from_numpy(is_unused)].mean(0)
    score = torch.nn.functional.cosine_similarity(without_pc, reference[None, :]).numpy()
    threshold = np.quantile(score[is_unused], 0.05)
    flagged = (score >= threshold) & ~is_unused & ~is_special
    print("undertrained threshold", round(float(threshold), 4), "flagged non-unused", flagged.sum())
    return score, flagged


def wordnet_lemmas():
    import nltk

    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet

    return set(wordnet.all_lemma_names())


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tok = gemma.tokenizer()
    key, W = gemma.embedding()
    n_vocab = min(len(W), len(tok))
    W = W[:n_vocab]
    print("embedding", key, W.shape, "tokenizer", len(tok))

    pieces, is_unused, is_special = gemma.token_masks(tok, n_vocab)
    is_byte = np.array([is_byte_piece(p) for p in pieces])
    print("unused", is_unused.sum(), "special", is_special.sum(), "byte", is_byte.sum())
    undertrained_score, is_undertrained = undertrained(W, is_unused, is_special)

    lemmas = wordnet_lemmas()
    bare = [p.replace("▁", " ") for p in pieces]
    core = [b.strip() for b in bare]
    scripts = np.array([script_of(p if is_byte_piece(p) else b) for p, b in zip(pieces, bare)])
    is_alpha = np.array([c.isalpha() for c in core])
    word_initial = np.array([p.startswith("▁") for p in pieces])
    is_word = is_alpha & word_initial & np.array([len(c) >= 2 for c in core])
    in_wordnet = np.array([a and c.lower() in lemmas for a, c in zip(is_alpha, core)])

    kept = np.flatnonzero(~is_unused & ~is_special)
    print("kept", len(kept), "of", n_vocab)
    np.save(out / "emb.npy", W[kept])
    np.save(out / "token_ids.npy", kept)
    (out / "tokens.json").write_text(json.dumps([gemma.show(pieces[i]) for i in kept], ensure_ascii=False))
    np.savez(
        out / "features.npz",
        script=scripts[kept], is_alpha=is_alpha[kept], is_word=is_word[kept], in_wordnet=in_wordnet[kept],
        is_byte=is_byte[kept], undertrained_score=undertrained_score[kept], is_undertrained=is_undertrained[kept],
        norm=np.linalg.norm(W[kept], axis=1),
    )
    for name, mask in [("word", is_word), ("wordnet", in_wordnet), ("undertrained", is_undertrained)]:
        print(name, int(mask[kept].sum()))
    names, counts = np.unique(scripts[kept], return_counts=True)
    print("scripts", [(str(n), int(c)) for n, c in sorted(zip(names, counts), key=lambda x: -x[1])[:12]])
