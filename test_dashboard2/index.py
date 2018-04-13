import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app, server
from apps import commercial_beer_recommender, commercial_beer_recommender_rating_filter, commercial_beer_recommender_custom, commercial_beer_viz_styles, commercial_beer_viz_breweries


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
])


index_page = html.Div([
    dcc.Link('Beer Recommender', href='/apps/beer_recommender'),
    html.Br(),
    dcc.Link('Beer Recommender with Filter', href='/apps/beer_recommender_w_filter'),
    html.Br(),
    dcc.Link('Custom Beer Recommender', href='/apps/beer_recommender_custom'),
    html.Br(),
    dcc.Link('Explore the World of Beers (by style)', href='/apps/commercial_beer_viz_styles'),
	html.Br(),
    dcc.Link('Explore the World of Beers (by brewery)', href='/apps/commercial_beer_viz_breweries')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/app1':
        return app1.layout
    elif pathname == '/apps/app2':
        return app2.layout
    elif pathname == '/apps/beer_recommender':
	    return commercial_beer_recommender.layout
    elif pathname == '/apps/beer_recommender_w_filter':
        return commercial_beer_recommender_rating_filter.layout
    elif pathname == '/apps/beer_recommender_custom':
        return commercial_beer_recommender_custom.layout
    elif pathname == '/apps/commercial_beer_viz_styles':
        return commercial_beer_viz_styles.layout
    elif pathname == '/apps/commercial_beer_viz_breweries':
        return commercial_beer_viz_breweries.layout
    else:
        return index_page

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')