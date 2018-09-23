
import json

import parse


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


def load_texts(refs, missing, window=100, **kwargs):
    missed = 0
    for fname, tree in parse.parse_dir(**kwargs):
        for seg, note in parse.get_seg_notes(tree):
            try:
                url, lang = parse.get_ptr(note)   # throws ValueError if not correct seg
                prev, post = parse.get_context(seg, window=window)
                in_missing = url in missing  # flag to check if missing
                yield {'id': note.attrib[parse.add_ns('id', ns='w3')],
                       'url': url,
                       'lang': lang,
                       'ref': refs[url]['target'],
                       'type': parse.get_link_type(parse.get_link(note)),
                       'text': ' '.join(w['word'] for w in parse.get_text(seg)),
                       'textdata': list(parse.get_text(seg)),
                       'textcontext': {'prev': prev, 'next': post},
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

    with open('SCT1-5.json', 'w') as f:
        for text in load_texts(refs, missing, dirname='../biblindex/SCT1-5'):
            f.write("{}\n".format(json.dumps(text)))
