
from flask import Flask, Response, current_app
from json import dumps
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config.from_object('config')

model = load_model('./model.h5')


@app.route('/')
def predict():
    return Response(
        dumps({
            'error': None,
            'data': model.predict(True)
        }),
        status=200,
        content_type='application/json'
    )
