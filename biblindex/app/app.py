
import os

import bottle
from routes import build_website


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    STATIC_ROOT = os.path.join(PROJECT_ROOT, 'static').replace('\\', '/')

    app = build_website(PROJECT_ROOT)

    @app.route('/static/<filepath:path>')
    def server_static(filepath):
        return bottle.static_file(filepath, root=STATIC_ROOT)

    bottle.run(app=app, reloader=args.dev, debug=args.dev, port=args.port)
