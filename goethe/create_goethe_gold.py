
"""
Here is the link to the data. You really only need the file
jgoethe.xml, the entities and the schema, but I was to lazy to sort it
out. The file contains the works of young Goethe (basically all he
wrote up to his 26th birthday) and our commentary and additional
texts, for example a mythological dictionary and the Luther bibel. All
references to the bibel are internal links but it doesn't always say
explicitly bibel in the notes text. So if you want to extract *all*
references, the safest way is probably to collect all ref target ids
in the bibel and then search for them in the rest of the corpus.
"""

import re
import string
import collections
import os
from lxml import etree
import tqdm

ORIGINAL_PATH = './data/jgoethe.xml'
PATH = './data/jgoethe.utf8.noents.xml'
PARSER = etree.XMLParser()


def bs4_transform_entities(path=ORIGINAL_PATH, output=PATH):
    import bs4
    with open(path, encoding='iso-8859-1') as f:
        # ignore first 10 lines (entities and stuff)
        for _ in range(10):
            next(f)
        t = bs4.BeautifulSoup(f.read(), 'lxml').find('tei.2')
    with open(output, 'w') as f:
        f.write(str(t))


if not os.path.isfile(PATH):
    bs4_transform_entities()


def get_bible_verses(tree):
    # # check all in bible
    # bible = tree.xpath('//div1[@id="JG46062"]')[0]
    # for verse in tqdm.tqdm(verses, total=len(verses)):
    #     assert bible.xpath('//*[@id="{}"]'.format(verse['verse_id'])), verse['verse_id']
    verses = []
    for verse in tree.xpath('//l[@rend="Bibelvers"]'):
        text = verse.find('anchor').tail.strip()
        verses.append({
            "anchor": verse.find('anchor').attrib['id'],
            "verse_id": verse.attrib['id'],
            "verse_num": int(verse.attrib['n']),
            "text": text
        })

    return verses


def remove_bible_verses(tree):
    bible = tree.xpath('//div1[@id="JG46062"]')[0]
    bible.getparent().remove(bible)


# look for references in the tree
def read_refs(path='refs.txt'):
    with open(path) as f:
        refs = list(set([line.strip() for line in f]))
    return refs


def extract_ref_text(text, note):
    # extract actual text
    if not note.strip().endswith(']'):
        return

    # normalize whitespace
    text, note = ' '.join(text.strip().split()), ' '.join(note.strip().split())
    note = note[:-1]
    if note.startswith('['):
        note = note[1:]
    if '[...]' in note:
        start, *_, end = note.split('[...]')
        start, end = start.strip(), end.strip()
        if not end:
            return start

        if start not in text or end not in text:
            print(repr(text))
            return text

        start_index = text.index(start)
        end_index = text.index(end)

        return text[start_index: end_index + len(end)]

    else:
        return note


def get_ref_text(ref):
    p = ref.getparent()

    if p.getparent() is None:
        return
    if p.find('hi') is None:
        return

    if p.getprevious() is not None:
        text = p.getprevious().tail
    else:
        text = ''
        if p.getparent().getprevious() is not None:
            text += p.getparent().getprevious().text or ''
        text += p.getparent().text
        if p.getparent().getnext() is not None:
            text += p.getparent().getnext().text or ''

    note = p.find('hi').text

    return extract_ref_text(text, note)


def get_dataset(tree):
    targets = set(read_refs())
    refs, errors, nonote = [], [], []
    by_anchor = collections.defaultdict(list)

    # get verses
    verses = get_bible_verses(tree)

    # remove verses to avoid getting bible verses as matches
    remove_bible_verses(tree)

    for verse in tqdm.tqdm(verses):
        if verse['anchor'] not in targets:
            continue
        for ref in tree.xpath('//ref[@target="{}"]'.format(verse['anchor'])):
            if ref.getparent().tag != 'note':
                nonote.append((verse, ref))
                continue

            ref_text = get_ref_text(ref)
            if not ref_text:
                errors.append((verse, ref))
                continue

            if verse['anchor'] in by_anchor and ref_text in by_anchor[verse['anchor']]:
                print("Duplicate")
                continue

            refs.append(
                {'anchor': verse['anchor'],
                 'target': verse,
                 'ref': ref,
                 'source': ref_text})

            by_anchor[verse['anchor']].append(refs[-1]['source'])

    return refs, errors, nonote


regex = re.compile('(^[{chars}]|[{chars}]+$)'.format(
    chars=''.join(re.escape(c) for c in string.punctuation)))


def strip_punct(s):
    return [re.sub(regex, '', w) for w in s]


if __name__ == '__main__':
    tree = etree.parse(PATH, PARSER)
    bible = get_bible_verses(tree)
    refs, errors, nonote = get_dataset(tree)

    # remove single word references: "wie Kain"
    for r in refs:
        if len(r['source'].split()) < 3:
            print(r)

    refs = [r for r in refs if len(r['source'].split()) > 2]

    with open('goethe-gold.csv', 'w') as f:
        for ref in refs:
            sid1, sid2 = '_', ref['anchor'] + '-' + ref['target']['verse_id']
            s1 = ref['source']
            s2 = ref['target']['text']
            s1 = ' '.join(strip_punct(s1.split()))
            s2 = ' '.join(strip_punct(s2.split()))
            l1, l2 = '_', '_'
            f.write('\t'.join([sid1, sid2, s1, s2, l1, l2]) + '\n')

    anchors = set([ref['anchor'] for ref in refs])
    with open('goethe-background.csv', 'w') as f:
        for verse in bible:
            if verse['anchor'] in anchors:
                print("Duplicate")
                continue
            text = ' '.join(strip_punct(verse['text'].split()))

            sid = verse['anchor'] + '-' + verse['verse_id']
            f.write('\t'.join([sid, text, '_']) + '\n')

# for ref in refs:
#     print("ANCHOR:", ref["anchor"])
#     print("target ->", ref["target"]["text"])
#     print("source ->", ref['source'])
#     print()
#     print()


# mrefs = [r for r in refs if r['anchor'] == 'AMOSBHAB']


# for r in refs:
#     if not r['target_note'].strip().endswith(']'):
#         print(r['target_note'])
