import os
from flask import Flask, session, redirect, url_for  # Import the redirect and url_for functions
from flask_login import LoginManager, current_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_dropzone import Dropzone
from flask_session import Session

db = SQLAlchemy()
DB_NAME = "database.db"

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    
    db.init_app(app)

    from .views import views
    from .auth import auth
    from .code import code

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')
    app.register_blueprint(code, url_prefix='/')

    from .models import User, Note, Drugs, Info
    
    with app.app_context():
      #  db.drop_all()
        db.create_all()

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))
    
    @app.before_request
    def ensure_device_seen():
        if 'device_seen' not in session and current_user.is_authenticated:
            logout_user()
            return redirect(url_for('auth.login'))  # Adjust 'auth.login' as necessary

    return app


def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        
        print('Created Database!')
