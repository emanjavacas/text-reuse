
N=(1000 5000 10000 20000 35000);
for n in ${N[@]}; do
    # Retrieval with word embeddings
    python retrieval.py --n_background $n --lemma
    python retrieval.py --n_background $n
    # Retrieval with word embeddings on non-lexical matches at 2
    python retrieval.py --avoid_lexical --n_background $n --lemma
    python retrieval.py --avoid_lexical --n_background $n
    # Tesserae
    python baseline.py --n_background $n --lemma
    python baseline.py --n_background $n
done
