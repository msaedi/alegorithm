import flask

server = flask.Flask(__name__)
server.secret_key = 'very secret'

@server.route('/')
def home(name=None):
    return 'Hello, World!'