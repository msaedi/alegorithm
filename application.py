# W210 Course Project
# Alegorithm

import sqlite3
import dateutil.parser as dateparser
import io
import csv
import math
import os
from flask import Flask, Response, abort, render_template, send_from_directory, session, request
from flask import Flask, flash, redirect, render_template, request, session, abort
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from scipy.stats.stats import pearsonr
import numpy as np
import pandas as pd
import StringIO
import database_access as dbHandler

app = Flask(__name__, static_url_path='')
app.secret_key = 'very secret'
api = Api(app)
CORS(app)


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

class Login(Resource):
    def get(self):
        print('Login API')
        parser = reqparse.RequestParser()
        parser.add_argument('username', type=str)
        parser.add_argument('password', type=str)
        args = parser.parse_args()

        print(args.username)
        print(args.password)
        cnx = sqlite3.connect('database/beer_data.db')
        user = pd.read_sql_query("SELECT * FROM Users WHERE username='{}'".format(args.username), con=cnx)
        cnx.close()
        print(user)
        name = ""
        if not user.empty:
            if args.password == user.PASSWORD[0]:
                print 'success'
                response = 'success'
                name = user.NAME[0]
                session['logged_in'] = True
            else:
                print 'Incorrect passwrod'
                flash('wrong password!')
                response = 'fail'
        else:
            print 'No such username'
            response = 'error'

        dest = StringIO.StringIO()
        dest.write("msg\tname\n")
        dest.write(response+'\t'+name+'\n')
        print dest.getvalue()

        return Response(dest.getvalue(), mimetype="text")


class Register(Resource):
    def get(self):
        print('Registration API')
        parser = reqparse.RequestParser()
        parser.add_argument('name', type=str)
        parser.add_argument('email', type=str)
        parser.add_argument('username', type=str)
        parser.add_argument('password', type=str)
        args = parser.parse_args()


        print(args.name)
        print(args.email)
        print(args.username)
        print(args.password)
        cnx = sqlite3.connect('database/beer_data.db')

        user = pd.read_sql_query("SELECT * FROM Users WHERE username='{}'".format(args.username), con=cnx)
        username = args.username
        print(user)
        if user.empty:
            print 'user added'
            response = 'user added'
            c = cnx.cursor()
            c.execute("INSERT INTO USERS (USERNAME, NAME, PASSWORD, EMAIL) VALUES ('{username}', '{name}', '{password}', '{email}')".format(
            username=args.username, name=args.name, password=args.password, email=args.email))
            cnx.commit()
            cnx.close()
        else:
            print 'user already exists'
            response = 'user exists'

        dest = StringIO.StringIO()
        dest.write("msg\tusername\n")
        dest.write(response+'\t'+username +'\n')
        print dest.getvalue()
        return Response(dest.getvalue(), mimetype="text")

api.add_resource(HelloWorld, '/hello')
api.add_resource(Login, '/login')
api.add_resource(Register, '/register')


@app.route('/')
@app.route('/index.html')
def index(name=None):
    print session.get('logged_in')
    if not session.get('logged_in'):
        return render_template('signin.html', name=name)
    else:
        return render_template('index.html', name=name)

@app.route('/signin.html', methods=['POST', 'GET'])
def signin(name=None):
    #session['logged_in'] = False
    return render_template('signin.html', name=name)

@app.route('/signup.html', methods=['POST', 'GET'])
def signup(name=None):
    return render_template('signup.html', name=name)
    
@app.route('/signout.html', methods=['POST', 'GET'])
def signout(name=None):
    session['logged_in'] = False
    return index()
    
@app.route('/assets/<path:path>')
def send_assets(path):
    return send_from_directory('assets', path)

@app.route('/data/<path:path>')
def send_data(path):
    return send_from_directory('data', path)


if __name__ == '__main__':
    app.secret_key = 'very secret'
    app.run(debug=False, host='0.0.0.0')
