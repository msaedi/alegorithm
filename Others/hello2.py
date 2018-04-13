import os
import flask
import dash
import dash_core_components as dcc
import dash_html_components as html

server = flask.Flask(__name__)
server.secret_key = 'very secret'

app = dash.Dash(__name__, server=server)


app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.layout = html.Div([
    html.H2('Hello World'),
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': i, 'value': i} for i in ['LA', 'NYC', 'MTL']],
        value='LA'
    ),
    html.Div(id='display-value')
])

@app.callback(dash.dependencies.Output('display-value', 'children'),
              [dash.dependencies.Input('dropdown', 'value')])
def display_value(value):
    return 'You have selected "{}"'.format(value)

@server.route('/')
def home(name=None):
    return 'Hello, World!'
	
@server.route('/dash')
def home(name=None):
	# call the dash app
    return 'Hello, World!'

if __name__ == '__main__':
    app.secret_key = 'very secret'
    app.run_server(debug=True, host='0.0.0.0')
