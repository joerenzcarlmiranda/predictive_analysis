from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Hybrid Model API Running'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']  # Format: [[1, 64], [2, 48], ...]

        # Split X and y
        X = np.array([row[0] for row in data]).reshape(-1, 1)
        y = np.array([row[1] for row in data])

        # Train both models
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        lr = LinearRegression()

        rf.fit(X, y)
        lr.fit(X, y)

        last_x = X[-1][0]

        # Predict future points
        periods = {
            'week': last_x + 0.25,
            'month': last_x + 1,
            'quarter': last_x + 3,
            'year': last_x + 12
        }

        hybrid_results = {}

        for key, future_x in periods.items():
            rf_pred = rf.predict([[future_x]])[0]
            lr_pred = lr.predict([[future_x]])[0]

            # Weighted average: 70% Random Forest + 30% Linear Regression
            hybrid_pred = 0.7 * rf_pred + 0.3 * lr_pred
            hybrid_results[key] = round(hybrid_pred, 2)

        return jsonify(hybrid_results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
