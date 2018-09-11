"""
Routes and views for the bottle application.
"""

import os
import collections
import json
from datetime import datetime

from tinydb import TinyDB, Query
from bottle import Bottle, view, request


# load resources
# - DB
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

    done = db.search(Query().source == path)

    if only_ids:
        return set([doc['id'] for doc in done])

    return done


def get_annotation_data(path):
    output = collections.defaultdict(list)

    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj['type'] == 'inexactQuotation-allusion':
                output[obj['sourceXml']].append(obj)

    return output


def get_annotation_stats(data):
    stats = []

    for sourceXml, items in data.items():
        annotated = db.search(Query().sourceXml == sourceXml)
        stats.append((sourceXml, len(items), len(annotated), len(annotated)//len(items)))

    return stats


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
            return dict(year=datetime.now().year,
                        root=root, opts=opts, path=None, stats=None)
        try:
            stats = get_annotation_stats(get_annotation_data(path))
            return dict(year=datetime.now().year,
                        root=root, opts=opts, path=path, stats=stats)
        except Exception:
            return dict(year=datetime.now().year,
                        root=root, opts=opts, path=path, stats=None)

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
                data = get_annotation_data(path)
            except Exception:
                pass
            finally:
                if data is not None:
                    done = retrieve_done()
                    data = [item for coll in data.values() for item in coll
                            if item['id'] not in done]
                    total = len(done) + len(data)

        return dict(year=datetime.now().year, path=path, data=data, total=total)

    @app.route('/review')
    @view('annotate')
    def review():
        """Review page."""
        return dict(year=datetime.now().year)

    return app
