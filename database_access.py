import sqlite3 as sql

def insertUser(username,password):
	con = sql.connect("database/beer_data.db")
	cur = con.cursor()
	cur.execute("INSERT INTO users (username,password) VALUES (?,?)", (username,password))
	con.commit()
	con.close()

def retrieveUsers():
	con = sql.connect("beer_data.db")
	cur = con.cursor()
	cur.execute("SELECT username, password FROM users")
	users = cur.fetchall()
	con.close()
	return users

def authenticateUser(username, password):
	con = sql.connect("beer_data.db")
	cur = con.cursor()
	cur.execute("SELECT username, password FROM user")
	users = cur.fetchall()
	con.close()
	print users
	return users
