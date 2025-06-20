from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow Laravel or frontend to connect

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'message': 'Predictive Analysis API is running',
        'status': 'healthy'
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']  # Expects: [[1, 100], [2, 120], ...]

        X = np.array([row[0] for row in data]).reshape(-1, 1)
        y = np.array([row[1] for row in data])

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        last_x = X[-1][0]

        prediction = {
            'week': round(model.predict([[last_x + 0.25]])[0], 2),
            'month': round(model.predict([[last_x + 1]])[0], 2),
            'quarter': round(model.predict([[last_x + 3]])[0], 2),
            'year': round(model.predict([[last_x + 12]])[0], 2)
        }

        return jsonify(prediction)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
