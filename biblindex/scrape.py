
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

# configure logger
# create logger with 'spam_application'
logger = logging.getLogger('scrape')
logger.setLevel(logging.INFO)
# create file handler which logs even debug messages
fh = logging.FileHandler('scrape.log')
fh.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)


def download_target(url):
    res = requests.get(url, headers={'User-Agent': UA})

    elem = etree.HTML(res.text).cssselect('span.label')

    if len(elem) < 1:
        logger.warn("Couldn't find any span.label for page [{}]".format(url))
        return

    if len(elem) > 1:
        logger.warn("Found more than one span.label for page [{}]".format(url))
        return

    elem = elem[0]

    return ''.join(elem.itertext())


if __name__ == '__main__':

    with open('register.txt', 'a+') as f:
        register = set([l.strip() for l in f])

    try:
        with open('scraped.json', 'a') as f:
            for link in get_links(dirname='../biblindex/SCT1-5/'):

                if link['url'] in register:
                    logger.info("Skipping [{}]; already downloaded".format(link['url']))
                    continue

                logger.info("Downloading [{}]".format(link['url']))
                target = download_target(link['url'])
                obj = {'url': link['url'], 'target': target}
                f.write('{}\n'.format(json.dumps(obj)))
                f.flush()           # force write immediatly

                register.add(obj['url'])

                # sleep 7,5 secs on average (6487 * 5 = 13,51 hours)
                sleep = random.randint(5, 10)
                logger.info("Sleeping {} secs".format(sleep))
                time.sleep(sleep)

    except KeyboardInterrupt:
        with open('register.txt', 'w') as f:
            for item in register:
                f.write('{}\n'.format(item))
