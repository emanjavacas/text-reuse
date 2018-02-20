
import time
import random
import logging
import json
import requests

from lxml import etree
from fake_useragent import UserAgent

from parse import get_links


# grab a user agent
UA = UserAgent().ie

logger = logging.getLogger('scrape')
logger.setLevel(logging.DEBUG)


def download_target(url):
    res = requests.get(url, headers={'User-Agent': UA})

    elem = etree.HTML(res.text).cssselect('span.label')

    if len(elem) < 1:
        logger.info("Couldn't find any span.label for page [{}]".format(res.text))
        return

    if len(elem) > 1:
        logger.info("Found more than one span.label for page [{}]".format(res.text))
        return

    elem = elem[0]

    return ''.join(elem.itertext())


if __name__ == '__main__':

    with open('scraped.json', 'w+') as f:
        for link in get_links(dirname='../biblindex/SCT1-5/'):
            target = download_target(link['url'])
            obj = {'url': link['url'], 'target': target}
            f.write('{}\n'.format(json.dumps(obj)))

            # sleep 7,5 secs on average (6487 * 5 = 13,51 hours)
            time.sleep(random.randint(5, 10))
