
import re
import os
import warnings

from lxml import etree


NSMAP = {'tei': 'http://www.tei-c.org/ns/1.0'}
NSREG = r'{[^}]+}'


def remove_ns(tag):
    return re.sub(NSREG, '', tag)


def add_ns(tag):
    return '{{{}}}{}'.format(NSMAP['tei'], tag)


def parse_file(fname):
    # /TEI/text/body/div[type="work"]/div[type="chapter" & n="1-..."]/
    # /p[style="txt_Normal"]
    with open(fname, 'r+') as f:
        tree = etree.fromstring(f.read().encode('utf-8'))

    return tree


def parse_dir(dirname='./biblindex/SCT1-5/'):
    for f in os.listdir(dirname):
        try:
            yield os.path.join(dirname, f), parse_file(os.path.join(dirname, f))
        except:
            warnings.warn("Couldn't parse [{}]".format(f))


def get_p(tree):
    for p in tree.xpath('//tei:p', namespaces=NSMAP):
        yield p


def get_seg_notes(tree):
    pair = []
    for e in tree.xpath('//tei:p[@style="txt_Normal"]/*', namespaces=NSMAP):
        if remove_ns(e.tag) == 'seg':
            pair.append(e)
        elif remove_ns(e.tag) == 'note':
            if len(pair) == 0:
                warnings.warn('<seg> followed by non <note>')
                pair = []
                continue

            pair.append(e)
            yield pair
            pair = []


def get_link_type(link_attrib):
    # there is a case where type is "inexactQuotation inexactQuotation"
    t = link_attrib.get('type').split()[0]
    allusion = link_attrib.get('ana')
    if allusion is not None:
        t += '-{}'.format('allusion')
    return t


def get_links(**kwargs):
    for fname, tree in parse_dir(**kwargs):
        for idx, note in enumerate(tree.xpath('//tei:p/tei:note', namespaces=NSMAP)):
            # get link
            link = note.find(add_ns('link'))

            # get ptr
            ptr_tag = '/'.join([add_ns('seg'), add_ns('bibl'), add_ns('ptr')])
            ptr = note.find(ptr_tag)
            if ptr is not None:
                url, lang = ptr.attrib['target'], ptr.attrib['targetLang']
            else:
                warnings.warn('No link at {} - [{}]'.format(idx, fname))

            yield {'type': get_link_type(link.attrib), 'url': url, 'lang': lang}
