from flask import Flask, jsonify
from flask import request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from tensorflow.keras.models import load_model
import pickle
import joblib

app = Flask(__name__)

lr = joblib.load('linear_regression.pkl')

pr = joblib.load('polynomial_regression.pkl')

rf = joblib.load('random_forest.pkl')

xg = joblib.load('xgboost.pkl')

model_lstm = load_model("lstm_model.h5")

with open("arima_model.pkl", "rb") as f:
    arima_model_fit = pickle.load(f)


@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({'message': 'API is working!', 'status': 'success'})


@app.route('/clustering', methods=['GET'])
def clustering_endpoint():
    # Get query parameters with defaults
    start_date = request.args.get('start_date', None)
    end_date = request.args.get('end_date', None)
    city = request.args.get('city', None)
    k = request.args.get('k', 3, type=int)
    
    # Validate required parameters
    if not all([start_date, end_date, city]):
        return jsonify({
            'status': 'error',
            'message': 'Missing required parameters. Please provide start_date, end_date, and city.'
        }), 400
    
    df = pd.read_csv('clustering_data.csv')

    # Convert date columns to datetime if they aren't already
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])

    # Filter by date range
    df_filtered = df[(df['time'] >= pd.to_datetime(start_date)) & 
                    (df['time'] <= pd.to_datetime(end_date))]

    # Filter by city
    df_filtered = df_filtered[df_filtered['city'].str.lower() == city.lower()]

    # Check if we have data after filtering
    if len(df_filtered) == 0:
        return jsonify({
            'status': 'error',
            'message': f'No data found for city "{city}" between {start_date} and {end_date}'
        }), 404
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_filtered[['x', 'y']])
    df_filtered['cluster'] = kmeans.labels_

    data = []
    for i in range(len(df_filtered)):
        data.append({
            'time': df_filtered.iloc[i]['time'].strftime('%Y-%m-%d %H:%M:%S'),
            'x': df_filtered.iloc[i]['x'],
            'y': df_filtered.iloc[i]['y'],
            'cluster': int(df_filtered.iloc[i]['cluster'])
        })
        
    return jsonify({
        'status': 'success',
        'data': data,
    })

@app.route('/line', methods=['GET'])
def line_endpoint():
    # Get query parameters with defaults
    start_date = request.args.get('start_date', None)
    end_date = request.args.get('end_date', None)
    city = request.args.get('city', None)
    
    # Validate required parameters
    if not all([start_date, end_date, city]):
        return jsonify({
            'status': 'error',
            'message': 'Missing required parameters. Please provide start_date, end_date, and city.'
        }), 400
    
    df = pd.read_csv('line_data.csv')

    # Convert date columns to datetime if they aren't already
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])

    # Filter by date range
    df_filtered = df[(df['time'] >= pd.to_datetime(start_date)) & 
                    (df['time'] <= pd.to_datetime(end_date))]

    # Filter by city
    df_filtered = df_filtered[df_filtered['city'].str.lower() == city.lower()]

    # Check if we have data after filtering
    if len(df_filtered) == 0:
        return jsonify({
            'status': 'error',
            'message': f'No data found for city "{city}" between {start_date} and {end_date}'
        }), 404
    

    x_cords = df_filtered['time'].tolist()
    X = df_filtered.drop(columns=["demand","city","time"])
    y = df_filtered["demand"]

    
    print(X.shape)
    lr_pred = lr.predict(X)

    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    pr_pred = pr.predict(poly_features.transform(X))

    rf_pred = rf.predict(X)

    xg_pred = xg.predict(X)

    forecast = arima_model_fit.forecast(steps=len(X))

    X_test_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
    lstm_pred = model_lstm.predict(X_test_lstm)
    

    
    return jsonify({
        'status': 'success',
        'results': {
            'x': x_cords,
            'actual': y.tolist(),
            'lr_pred': lr_pred.tolist(),
            'pr_pred': pr_pred.tolist(),
            'rf_pred': rf_pred.tolist(),
            'xg_pred': xg_pred.tolist(),
            'arima_pred': forecast.tolist(),
            'lstm_pred': lstm_pred.flatten().tolist()
            }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
