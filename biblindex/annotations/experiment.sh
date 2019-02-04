
N=(1000 5000 10000 20000 35000);
for n in ${N[@]}; do
    # Retrieval with word embeddings
    python word_embeddings.py --n_background $n --lemma
    python word_embeddings.py --n_background $n
    # Word_Embeddings with word embeddings on non-lexical matches at 2
    python word_embeddings.py --avoid_lexical --n_background $n --lemma
    python word_embeddings.py --avoid_lexical --n_background $n
    # Lexical
    python lexical.py --n_background $n --lemma
    python lexical.py --n_background $n
    # Elmo
    python elmo.py --n_background $n
done
