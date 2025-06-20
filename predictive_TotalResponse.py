from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow Laravel or frontend to connect

# ✅ Health check endpoint for UptimeRobot
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'message': 'Predictive Analysis API is running',
        'status': 'healthy'
    }), 200

# ✅ Predict endpoint with week, month, quarter, year
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']  # Expects: [[1, 100], [2, 120], ...]

        # Prepare training data
        X = np.array([row[0] for row in data]).reshape(-1, 1)
        y = np.array([row[1] for row in data])

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        last_x = X[-1][0]

        # Predict future periods
        prediction = {
            'week': model.predict([[last_x + 0.25]])[0],
            'month': model.predict([[last_x + 1]])[0],
            'quarter': model.predict([[last_x + 3]])[0],
            'year': model.predict([[last_x + 12]])[0]
        }

        # Round results
        rounded = {key: round(val, 2) for key, val in prediction.items()}

        return jsonify(rounded)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
