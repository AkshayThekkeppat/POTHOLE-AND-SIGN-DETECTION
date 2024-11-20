from pothole import app
from pothole import db,app
from pothole import db,app,login_manager
from flask_login import UserMixin

@login_manager.user_loader
def load_user(id):
    return Register.query.get(int(id))



class Register(db.Model, UserMixin):
    
    id=db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(80))
    password = db.Column(db.String(80))
    usertype = db.Column(db.String(80))
    name = db.Column(db.String(80))
    contact = db.Column(db.String(80))
    address = db.Column(db.String(80))
    qualification = db.Column(db.String(80))
    experience = db.Column(db.String(80))
    status = db.Column(db.String(80),default='NULL')
    


class Complaints(db.Model, UserMixin):
    id=db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String(80))
    message = db.Column(db.String(80))
    uid = db.Column(db.String(80))
    response = db.Column(db.String(80))
    email = db.Column(db.String(80))
    status = db.Column(db.String(80),default="NULL")


class Speedlimit(db.Model, UserMixin):
    id=db.Column(db.Integer, primary_key=True)
    district = db.Column(db.String(80))
    area = db.Column(db.String(80))
    limit = db.Column(db.String(80))


class Rules(db.Model, UserMixin):
    id=db.Column(db.Integer, primary_key=True)
    rules = db.Column(db.String(80))
    image = db.Column(db.String(80))
    


class News(db.Model, UserMixin):
    id=db.Column(db.Integer, primary_key=True)
    news = db.Column(db.String(80))
    image = db.Column(db.String(80))

    
class Projects(db.Model, UserMixin):
    id=db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(80))
    image = db.Column(db.String(80))
    details = db.Column(db.String(80))


class Detection(db.Model, UserMixin):
    id=db.Column(db.Integer, primary_key=True)
    filename=db.Column(db.String(200))
    data=db.Column(db.String(200))

   