# W210 Course Project
# Alegorithm

from flask import Flask, flash, redirect, render_template, request, session, abort, send_from_directory
import os
from sqlalchemy.orm import sessionmaker
from database.tabledef import *
engine = create_engine('sqlite:///database/alegorithm.db', echo=True)

server = Flask(__name__)
server.secret_key = 'very secret'


@server.route('/')
@server.route('/<name>')
@server.route('/index.html')
def home(name=None):
    print 'name is {}'.format(name)
    print 'index logged in is {}'.format(session.get('logged_in'))
    if not session.get('logged_in'):
        return render_template('signin.html')
    else:
        print 'Session name is {}'.format(session.get('logged_in_name'))
        return render_template('index.html', name=session.get('logged_in_name'))
 
@server.route('/login', methods=['POST'])
def do_admin_login():
    
    POST_USERNAME = str(request.form['username'])
    POST_PASSWORD = str(request.form['password'])
	
    Session = sessionmaker(bind=engine)
    s = Session()
    query = s.query(User)
    query = s.query(User).filter(User.username.in_([POST_USERNAME]), User.password.in_([POST_PASSWORD]) )
    result = query.first()
    if result:
        session['logged_in'] = True
        session['logged_in_username'] = POST_USERNAME
        session['logged_in_name'] = result.name
        print 'logged in name is {}'.format(session.get('logged_in_name'))
    else:
        flash('wrong password!')
    return render_template('index.html', name=session['logged_in_name'])
 
@server.route("/logout")
def logout():
    session['logged_in'] = False
    session['logged_in_as'] = None
    session['logged_in_name'] = None
    return home()
	

@server.route("/signup")
def signup():
    return render_template('signup.html')
	
@server.route("/register", methods=['POST'])
def register():
    print 'register'	
    POST_USERNAME = str(request.form['username'])
    POST_PASSWORD = str(request.form['password'])
    POST_NAME = str(request.form['name'])
    POST_EMAIL = str(request.form['email'])
	
    Session = sessionmaker(bind=engine)
    s = Session()
    query = s.query(User)
    query = s.query(User).filter(User.username.in_([POST_USERNAME]))
    result = query.first()
    if result:
        print 'result is {}'.format(result.username)
        flash('user exists!')
        print '{} already exists'.format(POST_NAME)
        return signup()
    else:
        user = User(POST_USERNAME, POST_PASSWORD, POST_NAME, POST_EMAIL)
        s.add(user)
        s.commit()
        session['logged_in'] = True
        session['logged_in_as'] = POST_USERNAME
        session['logged_in_name'] = POST_NAME
        print '{} has been added to the database'.format(POST_NAME)
        return render_template('index.html', name=session.get('logged_in_name'))

@server.route("/select_beers")
def select_beers():
    if not session.get('logged_in'):
        print 'select beer logged in is {}'.format(session.get('logged_in'))
        print 'Beer selection by {}'.format(session.get('logged_in_name'))
        return render_template('signin.html')
    else:
        print 'Beer selection by {}'.format(session.get('logged_in_name'))
        beer_list = []
        Session = sessionmaker(bind=engine)
        s = Session()
        query = s.query(Beer).filter(Beer.style.ilike('%ipa%'), Beer.rating >= 3.5 ).order_by(Beer.rating.desc())
        for row in query.limit(10).all():
            print 'Beer: {n}, Rating: {r}'.format(n=row.name, r=row.rating)
            beer_list.append(row.name)
        return render_template('select_beers.html', beer_list = beer_list, name=session.get('logged_in_name'))
	
@server.route("/add_user_beers", methods=['POST'])
def add_user_beers():
    POST_BEERS = request.form.getlist('user_beers')
    beer_list = [str(beer) for beer in POST_BEERS]
    print beer_list
    return select_beers()
	
@server.route('/assets/<path:path>')
def send_assets(path):
    return send_from_directory('assets', path)

@server.route('/data/<path:path>')
def send_data(path):
    return send_from_directory('data', path)
