from werkzeug.wsgi import DispatcherMiddleware
from alegorithm import application as frontend
from dashboard.commercial_beer_recommender_rating_filter import application as backend

application = DispatcherMiddleware(frontend, {'/backend': backend})