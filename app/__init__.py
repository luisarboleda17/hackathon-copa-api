
from flask import Flask, Response, request, current_app
from json import dumps
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd

app = Flask(__name__)
app.config.from_object('config')

graph = tf.get_default_graph()
model = load_model('model.h5')


@app.route('/')
def predict():
    body = request.get_json()

    body = pd.DataFrame.from_dict(body, orient='index')
    current_app.logger.warn(body)

    input_ = body.drop(['was_overbooked', 'is_weekend', 'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
                        'DESTINATION_ORIGIN',
                        'Airport Origin_origin',
                        'Country ORIGIN_origin',
                        'Airport DESTINATION_origin',
                        'Country DESTINATION_origin',
                        'FESTIVO en ORIGEN_origin',
                        'FESTIVO en DESTINO_origin',
                        'cc'
                        ], axis=1)
    embedding = body.drop(['FESTIVO en ORIGEN', 'FESTIVO en DESTINO',
                           'DESTINATION_ORIGIN',
                           'Airport Origin_origin',
                           'Country ORIGIN_origin',
                           'Airport DESTINATION_origin',
                           'Country DESTINATION_origin',
                           'FESTIVO en ORIGEN_origin',
                           'FESTIVO en DESTINO_origin',
                           'cc'
                           ], axis=1)
    feature = body.drop(['FLIGHT_NUMBER', 'ORIGIN', 'DESTINATION',
                         'TotalAuthorized', 'TotalSeatSold', 'TotalSeatAvailable', 'DateMonth',
                         'DateYear', 'Airport ORIGIN', 'Country ORIGIN', 'Airport DESTINATION',
                         'Country DESTINATION', 'FESTIVO en ORIGEN', 'FESTIVO en DESTINO',
                         'hour', 'day',
                         'DESTINATION_ORIGIN',
                         'Airport Origin_origin',
                         'Country ORIGIN_origin',
                         'Airport DESTINATION_origin',
                         'Country DESTINATION_origin',
                         'FESTIVO en ORIGEN_origin',
                         'FESTIVO en DESTINO_origin',
                         'cc'
                         ], axis=1)
    with graph.as_default():
        prediction = model.predict([embedding, feature, input_])

    return Response(
        dumps({
            'error': None,
            'data': prediction
        }),
        status=200,
        content_type='application/json'
    )
