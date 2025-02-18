from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/regression', methods=['POST'])
def regression():
    data = request.json
    x = np.array(data.get("x", []), dtype=float)
    y = np.array(data.get("y", []), dtype=float)

    n = len(x)

    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - (np.sum(x)) ** 2)
    intercept = (np.sum(y) - slope * np.sum(x)) / n

    y_hat = slope * x + intercept

    residuals = y - y_hat

    sum_squared_residuals = np.sum(residuals ** 2)

    # Calculate sigma
    sigma = np.sqrt(sum_squared_residuals / (n - 2))

    return jsonify({
        "slope": slope,
        "intercept": intercept,
        "standard_error": sigma
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
