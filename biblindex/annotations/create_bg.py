
import os
import difflib
from lxml import etree
import utils


def LCS(a, b):
    return difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))


def get_doc_id(tree):
    return tree.find('//div1').attrib['n'] + '-' + tree.find('//div2').attrib['n']


def get_verses(tree):
    elems = tree.xpath('//*[local-name() = "s" or local-name() = "milestone"]')
    for idx, (milestone, s) in enumerate(zip(elems[::2], elems[1::2])):
        assert milestone.tag == 'milestone', milestone.tag
        assert s.tag == 's', s.tag
        assert idx + 1 == int(milestone.attrib['n']), (idx + 1, milestone.attrib['n'])
        yield idx + 1, s.text


if __name__ == '__main__':
    import pie
    model = pie.SimpleModel.load("./capitula.model.tar")
    bible = list(utils.load_bible().values())
    texts = []
    with open('background.bible.csv', 'w') as outf:
        for f in os.listdir('./NT'):
            if not f.endswith('xml'):
                continue
            print(f)

            with open(os.path.join('NT', f)) as fn:
                tree = etree.fromstring(fn.read().encode('utf-8')).getroottree()
                doc_id = get_doc_id(tree)

                try:
                    for idx, s in get_verses(tree):
                        text = s.replace('\n', ' ')
                        is_dup = False
                        for ref in bible:
                            ref = ref['text']
                            if LCS(text.split(), ref.split()).size / \
                               len(text.split()) > 0.9:
                                print("Dup!", f, idx)
                                print(text)
                                print(ref)
                                print()
                                is_dup = True
                                break
                        if not is_dup and text:
                            lemma = utils.lemmatize(model, text.lower().split())['lemma']
                            row = '\t'.join([doc_id+'-'+str(idx), text, ' '.join(lemma)])
                            outf.write(row + '\n')
                except Exception as e:
                    print('ERROR:', e)
