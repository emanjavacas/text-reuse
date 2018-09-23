
import re
import os
import warnings

from lxml import etree


NSMAP = {'tei': 'http://www.tei-c.org/ns/1.0',
         'w3': 'http://www.w3.org/XML/1998/namespace'}
NSREG = r'{[^}]+}'


# XML funcs
def remove_ns(tag):
    return re.sub(NSREG, '', tag)


def add_ns(tag, ns='tei'):
    return '{{{}}}{}'.format(NSMAP[ns], tag)


def parse_file(fname):
    # /TEI/text/body/div[type="work"]/div[type="chapter" & n="1-..."]/
    # /p[style="txt_Normal"]
    with open(fname, 'r+') as f:
        tree = etree.fromstring(f.read().encode('utf-8'))

    return tree


def parse_dir(dirname='./biblindex/SCT1-5/'):
    for f in os.listdir(dirname):
        if not f.endswith('xml'):
            warnings.warn("Ignoring file: {}".format(f))
            continue
        try:
            yield str(f), parse_file(os.path.join(dirname, f))
        except Exception:
            warnings.warn("Couldn't parse [{}]".format(f))


def get_links(**kwargs):
    for fname, tree in parse_dir(**kwargs):
        for idx, note in enumerate(tree.xpath('//tei:p/tei:note', namespaces=NSMAP)):
            # get link
            link = get_link(note)

            try:
                url, lang = get_ptr(note)
                yield {'type': get_link_type(link), 'url': url, 'lang': lang}

            except ValueError:
                warnings.warn('No link at {} - [{}]'.format(idx, fname))


# tree-level funcs
def get_p(tree):
    for p in tree.xpath('//tei:p', namespaces=NSMAP):
        yield p


def itercontext(seg, max_siblings, preceding):
    output = []
    for child in seg.itersiblings(preceding=preceding):
        if len(output) >= max_siblings:
            break
        if remove_ns(child.tag) == 'w':
            output.append(
                {'POS': child.attrib['ana'],
                 'lemma': child.attrib['lemma'],
                 'word': child.text})
        elif remove_ns(child.tag) == 'pc':
            output.append({'POS': 'PC', 'lemma': child.text, 'word': child.text})
        elif remove_ns(child.tag) == 'seg':
            children = list(child.iterdescendants())
            if preceding:
                children = reversed(children)
            for subchild in children:
                if remove_ns(subchild.tag) == 'w':
                    output.append(
                        {'POS': subchild.attrib['ana'],
                         'lemma': subchild.attrib['lemma'],
                         'word': subchild.text})
                elif remove_ns(subchild.tag) == 'pc':
                    output.append(
                        {'POS': 'PC', 'lemma': subchild.text, 'word': subchild.text})

    output = output[:max_siblings]
    if preceding:
        return output[::-1]
    return output


def get_context(elem, window):
    prev = itercontext(elem, window, True)
    post = itercontext(elem, window, False)
    return prev, post


def get_seg_notes(tree):
    pair, prev = [], None
    # txt_Normal: actual text
    elems = tree.xpath('//tei:p[@style="txt_Normal"]/*', namespaces=NSMAP)
    for idx, e in enumerate(elems):
        if remove_ns(e.tag) == 'seg':
            if len(pair) != 0:
                warnings.warn('<seg> followed by <seg>')
                pair = []
                continue
            pair.append(e)
            prev = idx

        elif remove_ns(e.tag) == 'note':
            if len(pair) == 0:
                warnings.warn('<seg> followed by non <note>')
                pair = []
                continue
            if idx != prev + 1:
                warnings.warn('<note> not immediatly following <seg>')
                pair = []
                continue

            pair.append(e)
            yield pair
            pair = []


# link-level funcs
def get_link_type(link):
    # there is a case where type is "inexactQuotation inexactQuotation"
    t = link.attrib.get('type').split()[0]
    allusion = link.attrib.get('ana')
    if allusion is not None:
        t += '-{}'.format('allusion')
    return t


# note-level funcs
def get_ptr(note):
    ptr = note.find('/'.join([add_ns('seg'), add_ns('bibl'), add_ns('ptr')]))
    if ptr is None:
        raise ValueError
    else:
        return ptr.attrib['target'], ptr.attrib['targetLang']


def get_link(note):
    return note.find(add_ns('link'))


# seg-level funcs
def get_text(seg):
    for item in seg.xpath('.//tei:w|.//tei:pc', namespaces=NSMAP):
        if remove_ns(item.tag) == 'w':
            yield {'POS': item.attrib['ana'],
                   'lemma': item.attrib['lemma'],
                   'word': item.text}
        else:
            yield {'POS': 'PC',
                   'lemma': item.text,
                   'word': item.text}
