from flask import Flask, request
import joblib
import numpy

MODEL_PATH1 = 'models/model1.pkl'
SCALER_X_PATH1 = 'models/scaler_x1.pkl'
SCALER_Y_PATH1 = 'models/scaler_y1.pkl'

MODEL_PATH2 = 'models/model2.pkl'
SCALER_X_PATH2 = 'models/scaler_x2.pkl'
SCALER_Y_PATH2 = 'models/scaler_y2.pkl'

app = Flask(__name__)

@app.route('/predict_price', methods=['GET'])
def predict():
    args = request.args
    model_choice = args.get('model_version', default=-1, type=int)

    floor = args.get('floor', type=int)
    open_plan = args.get('open_plan', type=int)
    rooms = args.get('rooms', type=int)
    studio = args.get('rooms', type=int)
    area = args.get("area", type=float)
    renovation = args.get('renovation', type=int)
    exposition_days = args.get('exposition_days', type=int)
    elite_apartment = args.get('exposition_days', type=int)

    if model_choice== 1:
        model1 = joblib.load(MODEL_PATH1)
        sc_x1 = joblib.load(SCALER_X_PATH1)
        sc_y1 = joblib.load(SCALER_Y_PATH1)

        x1 = numpy.array([floor, open_plan, rooms, studio, area, renovation]).reshape(1, -1)
        x1 = sc_x1.transform(x1)
        result1 = model1.predict(x1)
        result1 = sc_y1.inverse_transform(result1.reshape(1, -1))

        return str(result1[0][0])

    elif model_choice == 2:
        model2 = joblib.load(MODEL_PATH2)
        sc_x2 = joblib.load(SCALER_X_PATH2)
        sc_y2 = joblib.load(SCALER_Y_PATH2)

        x2 = numpy.array([floor, open_plan, rooms, studio, area, renovation, exposition_days, elite_apartment]).reshape(1, -1)
        x2 = sc_x2.transform(x2)
        result2 = model2.predict(x2)
        result2 = sc_y2.inverse_transform(result2.reshape(1, -1))

        return str(result2[0][0])

if __name__ == '__main__':
    app.run(debug=True, port=5444, host='0.0.0.0')