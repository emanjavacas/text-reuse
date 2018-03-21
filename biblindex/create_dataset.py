
import json

from parse import get_seg_notes, get_ptr, get_link, get_link_type, get_text, parse_dir


def load_scraped_refs(fname='scraped.json'):
    refs = []
    missing = set()
    with open(fname, 'r+') as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj['target'] is not None:
                refs.append(obj)
            else:
                missing.add(obj['url'])

    return {ref['url']: ref for ref in refs}, missing


def load_texts(refs, missing, **kwargs):
    missed = 0
    for fname, tree in parse_dir(**kwargs):
        for seg, note in get_seg_notes(tree):
            try:
                url, lang = get_ptr(note)
                in_missing = url in missing  # flag to check if missing
                link_type = get_link_type(get_link(note))
                text = ' '.join(w['word'] for w in get_text(seg))
                yield {'url': url,
                       'ref': refs[url]['target'],
                       'lang': lang,
                       'type': link_type,
                       'text': text,
                       'sourceXml': fname}
                in_missing = False
            except ValueError:
                continue
            except KeyError:
                missed += 1
                if not in_missing:
                    raise ValueError(
                        "Missing scraped text for ref not known to be missing")


if __name__ == '__main__':
    refs, missing = load_scraped_refs()

    with open('SCT1-5.json', 'a+') as f:
        for text in load_texts(refs, missing, dirname='../biblindex/SCT1-5'):
            f.write("{}\n".format(json.dumps(text)))
