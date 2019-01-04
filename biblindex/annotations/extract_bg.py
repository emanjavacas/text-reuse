
import json
import os
import difflib
from lxml import etree


def LCS(a, b):
    return difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))


def load_bible(path='./biblindex/splits/SCT1-5.json'):
    bible = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj['type'] == 'inexactQuotation-allusion':
                bible.append(obj['ref'])

    return bible


bible = load_bible()


texts = []
for f in os.listdir('./'):
    if not f.endswith('xml'):
        continue
    print(f)

    with open(f) as fn:
        tree = etree.fromstring(fn.read().encode('utf-8')).getroottree()

        for idx, s in enumerate(tree.findall('//s') or []):
            text = s.text.replace('\n', ' ')
            is_dup = False
            for ref in bible:
                if LCS(text.split(), ref.split()).size / len(text.split()) > 0.9:
                    print("Dup!", f, idx)
                    print(text)
                    print(ref)
                    print()
                    is_dup = True
                    break
            if not is_dup and text:
                texts.append(text)


with open('background.NT.txt', 'w') as f:
    for text in texts:
        f.write(text + '\n')
