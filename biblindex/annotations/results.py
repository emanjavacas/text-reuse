
import os
import glob
import utils
import pandas as pd

import utils


def read_csv(path, **kwargs):
    rows = []
    with open(path) as f:
        header = next(f).strip().split('\t')
        header = [int(field) if field.isdigit() else field for field in header]
        for line in f:
            rows.append(dict(zip(header, line.strip().split('\t')), **kwargs))

    for row in rows:
        for field in row:
            if isinstance(field, int):
                row[field] = float(row[field])

    return rows


def get_support(stopwords, avoid_lexical=True, lemmas=False, threshold=2):
    total = 0
    for s, t in zip(*utils.load_gold(lemmas=lemmas, stopwords=stopwords)):
        if avoid_lexical and len(set(s).intersection(set(t))) >= threshold:
            continue
        total += 1

    return total


bernard_stop = utils.load_stopwords('bernard.stop')

total_support = get_support(bernard_stop, avoid_lexical=False)

dataset = []
for f in glob.glob('./results/*csv'):
    approach, background, *_ = os.path.basename(f).split('.')
    lemmas = 'lemmas' in f
    background = int(background)
    extra = {
        'approach': approach,
        'lemmatized': lemmas,
        'background': background
    }

    if 'overlap' in f:
        continue

    rows = read_csv(f, **extra)
    dataset.extend(rows)

    # enrich with OOV scores
    if approach == 'lexical':

        enrich = read_csv('./results/distributional.{}.overlap{}.csv'.format(
            # threshold fixed at 2
            str(background) + '.lemmas' if lemmas else background, 2))

        support = get_support(bernard_stop, avoid_lexical=True, lemmas=lemmas)

        for row in rows:
            if not row['method'].startswith('tesserae'):
                continue

            for row2 in enrich:
                scores = {}
                for field in row:
                    if isinstance(field, int):
                        scores[field] = row[field] + row2[field] * (support/total_support)
                dataset.append({
                    'method': '{}-enriched-{}'.format(row['method'], row2['method']),
                    **extra,
                    **scores
                })


pd.DataFrame.from_dict(dataset).to_csv('results.csv')
