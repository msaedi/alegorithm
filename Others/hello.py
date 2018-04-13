import dash
import flask

server = flask.Flask(__name__)
server.secret_key = 'very secret'
app = dash.Dash(__name__, server=server)

@server.route('/')
def home(name=None):
    return 'Hello, World!'

if __name__ == '__main__':
    app.secret_key = 'very secret'
    app.run_server(debug=True, host='0.0.0.0')
