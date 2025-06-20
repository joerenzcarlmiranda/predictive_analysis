from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow Laravel or frontend to connect

# Health check endpoint for uptime monitor (e.g. UptimeRobot)
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

        # Split X and y
        X = np.array([row[0] for row in data]).reshape(-1, 1)
        y = np.array([row[1] for row in data])

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Predict next point (e.g., next month or quarter)
        next_x = np.array([[X[-1][0] + 1]])
        predicted = model.predict(next_x)[0]

        return jsonify({
            'next_period': int(next_x[0][0]),
            'predicted_value': round(predicted, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
