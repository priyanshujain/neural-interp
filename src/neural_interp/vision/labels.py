from functools import cache

GROUPS = {
    "cat": "feline.n.01",
    "dog": "dog.n.01",
    "vehicle": "wheeled_vehicle.n.01",
}


@cache
def _wordnet():
    import nltk

    try:
        from nltk.corpus import wordnet

        wordnet.synsets("cat")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet
    return wordnet


@cache
def synset_name(wnid: str) -> str:
    wn = _wordnet()
    return wn.synset_from_pos_and_offset("n", int(wnid[1:])).lemma_names()[0].replace("_", " ")


@cache
def group(wnid: str) -> str:
    wn = _wordnet()
    synset = wn.synset_from_pos_and_offset("n", int(wnid[1:]))
    ancestors = {s.name() for path in synset.hypernym_paths() for s in path}
    for label, root in GROUPS.items():
        if root in ancestors:
            return label
    return "other"
