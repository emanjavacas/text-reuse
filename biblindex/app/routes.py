"""
Routes and views for the bottle application.
"""

import os
import collections
import json
from datetime import datetime
import random

from tinydb import TinyDB, Query
from bottle import Bottle, view, request


db = TinyDB('./db.json')


def retrieve_path():
    path = db.search(Query().type == 'datapath')

    if path:
        return path[0]['path']

    return None


def retrieve_done(only_ids=True):
    path = retrieve_path()

    if not path:
        return []

    query = Query()
    done = db.search((query.path == path) & (query.type == 'annotation'))

    if only_ids:
        return set([doc['id'] for doc in done])

    return done


def get_annotation_data(path, nested=True):
    output = collections.defaultdict(list)

    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj['type'] == 'inexactQuotation-allusion':
                output[obj['sourceXml']].append(obj)

    if not nested:
        output = [item for coll in output.values() for item in coll]

    return output


def get_annotation_stats(data):
    stats = []

    for sourceXml, items in data.items():
        annotated = db.search(Query().sourceXml == sourceXml)
        percentage = round(100 * (len(annotated) / len(items)), 0)
        stats.append((sourceXml, len(items), len(annotated), percentage))

    return stats


def get_payload(active=True, **kwargs):
    return {
        "active": active,
        "year": datetime.now().year,
        **kwargs}


def build_website(root):
    app = Bottle()

    @app.route('/')
    @app.route('/home')
    @view('index')
    def home():
        """Renders the home page."""
        opts = [f for f in os.listdir(root) if f.endswith('json') and f != 'db.json']
        path = retrieve_path()

        if not path or os.path.basename(path) not in opts:
            return get_payload(active=False, root=root, opts=opts, path=None, stats=None)
        try:
            stats = get_annotation_stats(get_annotation_data(path))
            return get_payload(root=root, opts=opts, path=path, stats=stats)
        except Exception:
            return get_payload(active=False, root=root, opts=opts, path=path, stats=None)

    @app.route('/register', method='post')
    def register():
        path = request.params.get('path')
        dbpath = retrieve_path()

        if dbpath and dbpath == path:
            db.remove(Query().type == 'datapath')
        else:
            db.upsert({'type': 'datapath', 'path': path}, Query().type == 'datapath')

        return 'OK'

    @app.route('/annotate')
    @view('annotate')
    def annotate():
        """Annotate page."""
        path, data, total = retrieve_path(), None, None

        if path:
            try:
                data = get_annotation_data(path, nested=False)
            except Exception:
                pass
            finally:
                if data is not None:
                    done = retrieve_done()
                    data = [item for item in data if item['id'] not in done]
                    random.shuffle(data)
                    total = len(done) + len(data)

        return get_payload(path=path, data=data, total=total)

    @app.route('/saveAnnotation', method='post')
    def saveAnnotation():
        ann = dict(request.params)
        ann['timestamp'] = datetime.now().timestamp()
        ann['path'] = retrieve_path()
        ann['root'] = root
        ann['type'] = 'annotation'
        ann['prevSpan'] = int(ann['prevSpan'])
        ann['nextSpan'] = int(ann['nextSpan'])
        query = Query()
        db.upsert(ann, (query.id == ann['id']) & (query.path == ann['path']))
        return 'OK'

    @app.route('/review')
    @view('annotate')
    def review():
        """Review page."""
        path, data, total = retrieve_path(), None, None

        if path:
            try:
                data = get_annotation_data(path, nested=False)
            except Exception:
                pass
            finally:
                if data is not None:
                    done = {item['id']: item for item in retrieve_done(only_ids=False)}
                    data = [dict(item, **{'annotation': done[item['id']]})
                            for item in reversed(data) if item['id'] in done]
                    total = len(done) + len(data)

        return get_payload(path=path, data=data, total=total)

    return app
