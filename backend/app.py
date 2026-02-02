from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React to connect


# Global variable for loaded model
model = None
feature_columns = None


def load_model():
    """
    Load the trained model when server starts
    This runs ONCE, not on every request
    """
    global model, feature_columns
    
    try:
        print("Loading trained model...")
        model = joblib.load('../data/nba_model.pkl')
        
        # Define feature columns (same as in model.py)
        feature_columns = [
            'home_fg_pct', 'away_fg_pct',
            'home_fg3_pct', 'away_fg3_pct',
            'home_ft_pct', 'away_ft_pct',
            'home_reb', 'away_reb',
            'home_ast', 'away_ast',
            'home_stl', 'away_stl',
            'home_blk', 'away_blk',
            'home_tov', 'away_tov',
            'fg_pct_diff',
            'reb_diff',
            'ast_diff'
        ]
        
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check to verify server is running"""
    return jsonify({
        "status": "healthy",
        "message": "NBA Prediction API is running"
    })


@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Return information about the trained model"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_type": type(model).__name__,
        "accuracy": "88.27%",  # From your training results
        "features_used": len(feature_columns),
        "status": "ready"
    })


@app.route('/predict', methods=['POST'])
def predict_game():
    """
    Main prediction endpoint
    Expects JSON body with team stats
    Returns prediction with confidence
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Get data from request
    data = request.get_json()
    
    # Validate required fields
    required_fields = [
        'home_fg_pct', 'away_fg_pct',
        'home_fg3_pct', 'away_fg3_pct',
        'home_ft_pct', 'away_ft_pct',
        'home_reb', 'away_reb',
        'home_ast', 'away_ast',
        'home_stl', 'away_stl',
        'home_blk', 'away_blk',
        'home_tov', 'away_tov'
    ]
    
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400
    
    # Calculate difference features
    fg_pct_diff = data['home_fg_pct'] - data['away_fg_pct']
    reb_diff = data['home_reb'] - data['away_reb']
    ast_diff = data['home_ast'] - data['away_ast']
    
    # Create feature array in correct order
    features = [
        data['home_fg_pct'],
        data['away_fg_pct'],
        data['home_fg3_pct'],
        data['away_fg3_pct'],
        data['home_ft_pct'],
        data['away_ft_pct'],
        data['home_reb'],
        data['away_reb'],
        data['home_ast'],
        data['away_ast'],
        data['home_stl'],
        data['away_stl'],
        data['home_blk'],
        data['away_blk'],
        data['home_tov'],
        data['away_tov'],
        fg_pct_diff,
        reb_diff,
        ast_diff
    ]
    
    # Convert to numpy array with correct shape (1 row, 19 features)
    features_array = np.array([features])
    
    # Make prediction
    prediction = model.predict(features_array)[0]  # 0 or 1
    
    # Get prediction probability (confidence)
    prediction_proba = model.predict_proba(features_array)[0]
    
    # Format response
    if prediction == 1:
        winner = "home"
        confidence = prediction_proba[1] * 100  # Home win probability
    else:
        winner = "away"
        confidence = prediction_proba[0] * 100  # Away win probability
    
    response = {
        "prediction": winner,
        "confidence": round(confidence, 2),
        "home_win_probability": round(prediction_proba[1] * 100, 2),
        "away_win_probability": round(prediction_proba[0] * 100, 2)
    }
    
    return jsonify(response)


@app.route('/predict-simple', methods=['POST'])
def predict_simple():
    """
    Simplified prediction endpoint for testing
    Just takes team names and uses average stats
    """
    data = request.get_json()
    
    # Example with dummy data for testing
    # In a real app, you'd fetch actual team stats here
    
    home_team = data.get('home_team', 'Lakers')
    away_team = data.get('away_team', 'Celtics')
    
    # Use average NBA stats as example
    dummy_features = [
        0.46, 0.44,  # FG percentages
        0.36, 0.35,  # 3P percentages
        0.77, 0.75,  # FT percentages
        45, 43,      # Rebounds
        25, 24,      # Assists
        8, 7,        # Steals
        5, 5,        # Blocks
        14, 15,      # Turnovers
        0.02,        # FG diff
        2,           # Reb diff
        1            # Ast diff
    ]
    
    features_array = np.array([dummy_features])
    prediction = model.predict(features_array)[0]
    prediction_proba = model.predict_proba(features_array)[0]
    
    response = {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_winner": home_team if prediction == 1 else away_team,
        "confidence": round(max(prediction_proba) * 100, 2)
    }
    
    return jsonify(response)


# Main execution
if __name__ == '__main__':
    # Load model when server starts
    load_model()
    
    # Run Flask app
    print("Starting NBA Prediction API...")
    print("Server running on http://localhost:5000")
    app.run(debug=True, port=5000)