
import os
import re
import requests
import urllib
from lxml import etree
from bs4 import BeautifulSoup

TOC = "http://www.perseus.tufts.edu/hopper/xmltoc?doc=Perseus%3Atext%3A1999.02.0060"
toc = etree.fromstring(urllib.request.urlopen(TOC).read()).getroottree()

targets, isin = [], False
for chunk in toc.findall("chunk"):
    if chunk.attrib.get('n') == 'Matthew':
        isin = True

    if isin:
        targets.append(chunk.attrib['ref'])
    else:
        continue

    if chunk.attrib.get('n') == 'Apocalypse':
        isin = False


for target in targets:
    url = 'http://www.perseus.tufts.edu/hopper/text?doc=' + target
    try:
        html = BeautifulSoup(urllib.request.urlopen(url).read(), 'html.parser')
    except Exception as e:
        print("oops at ", url, str(e))
        continue
    name = html.find(name='span', attrs={'class': 'title'}).text
    name = '_'.join(name.split())
    for idx, chapter in enumerate(html.findAll('a', title=re.compile("chapter .*"))):
        filename = '{}.{}.xml'.format(name, idx + 1)
        if os.path.isfile(filename):
            continue
        print(filename)
        url = re.sub('%[^%]*verse.*', '', chapter.get('href'))
        url = 'http://www.perseus.tufts.edu/hopper/xmlchunk' + url
        try:
            urllib.request.urlretrieve(url, filename=filename)
        except Exception as e:
            print("oops at " + filename, url, str(e))
