import os
from datetime import datetime
from flask import Flask, abort, request, jsonify, g, url_for
#from flask_httpauth import HTTPBasicAuth
"""from flask_sqlalchemy import SQLAlchemy
from passlib.apps import custom_app_context as pwd_context
from itsdangerous import (TimedJSONWebSignatureSerializer
                          as Serializer, BadSignature, SignatureExpired)"""

import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
from keras.constraints import maxnorm
from keras.optimizers import SGD

# initialization
classifier = ""
app = Flask(__name__)
"""app.config['SECRET_KEY'] = 'cyrsis3'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True"""

# extensions
#db = SQLAlchemy(app)
#auth = HTTPBasicAuth()

#User table
"""class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(32))
    phone = db.Column(db.String(32))
    username = db.Column(db.String(32), index=True)
    password_hash = db.Column(db.String(64))

    def hash_password(self, password):
        self.password_hash = pwd_context.encrypt(password)

    def verify_password(self, password):
        return pwd_context.verify(password, self.password_hash)

    def generate_auth_token(self, expiration=600):
        s = Serializer(app.config['SECRET_KEY'], expires_in=expiration)
        return s.dumps({'id': self.id})

    @staticmethod
    def verify_auth_token(token):
        s = Serializer(app.config['SECRET_KEY'])
        try:
            data = s.loads(token)
        except SignatureExpired:
            return None    # valid token, but expired
        except BadSignature:
            return None    # invalid token
        user = User.query.get(data['id'])
        return user

@auth.verify_password
def verify_password(username_or_token, password):
    # first try to authenticate by token
    print ("Password Verification called")
    print("User Name or Token=",username_or_token)
    user = User.verify_auth_token(username_or_token)
    if not user:
        # try to authenticate with username/password
        user = User.query.filter_by(username=username_or_token).first()
        if not user or not user.verify_password(password):
            return False
    g.user = user
    return True

#create new users
@app.route('/api/users', methods=['POST'])
def new_user():
    username = request.json.get('username')
    password = request.json.get('password')
    name = request.json.get('name')
    phone = request.json.get('phone')
    print ("User Registration Request Received")
    print (username)
    print (password)
    if username is None or password is None:
        return jsonify({'status': 'failure' , 'code': 400,})
        abort(400)    # missing arguments
    if User.query.filter_by(username=username).first() is not None:
        return jsonify({'status': 'failure' , 'code': 400,})
        abort(400)    # existing user
    user = User(username=username,name=name,phone=phone)
    user.hash_password(password)
    db.session.add(user)
    db.session.commit()
    return (jsonify({'status': 'success' , 'code': 200,}))
"""
@app.route('/api/v1/object_reg/',methods=['GET'])
def display_class():
    # load json and create model
    path = request.args.get('path')
    print (path)
    from keras.models import model_from_json
    from keras.optimizers import SGD
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    epochs = 25
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img('C:\\Users\\Aayush\\Desktop\\cat_or_dog_1.jpg', target_size = (32, 32))
    test_image = image.img_to_array(test_image)
    test_image = np.transpose(test_image, (2,0,1))
    test_image = np.expand_dims(test_image, axis = 0)
    result = loaded_model.predict(test_image)

    # search for the class
    for i in range(0,10):
        if result[0][i] == 1.0:
            index = i
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return (classes[index])

#main screen
@app.route('/')
def home():
    return ("Object recognition API")


if __name__ == '__main__':
    """db.create_all()
    if not os.path.exists('db.sqlite'):
        db.create_all()"""
    app.run(debug=True,host='0.0.0.0',port=3000)