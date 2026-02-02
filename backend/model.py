# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # for saving models


def load_processed_data(filepath):
    """Load the processed game data"""
    print("Loading processed data...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} games")
    return df


def prepare_features(df):
    """
    Select which columns to use as features for prediction
    """
    
    # Define feature columns (stats that help predict wins)
    feature_columns = [
        # Shooting percentages
        'home_fg_pct', 'away_fg_pct',
        'home_fg3_pct', 'away_fg3_pct',
        'home_ft_pct', 'away_ft_pct',
        
        # Rebounds, assists, etc.
        'home_reb', 'away_reb',
        'home_ast', 'away_ast',
        'home_stl', 'away_stl',
        'home_blk', 'away_blk',
        'home_tov', 'away_tov',
        
        # Calculated differences
        'fg_pct_diff',
        'reb_diff',
        'ast_diff'
    ]
    
    # Extract features (X) and target (y)
    X = df[feature_columns]
    y = df['home_win']  # What we're predicting (1 = home win, 0 = away win)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y, feature_columns


def split_data(X, y, test_size=0.2):
    """
    Split data into training and testing sets
    test_size=0.2 means 20% for testing, 80% for training
    """
    
    print("Splitting data into train/test sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,  # For reproducibility
        stratify=y  # Keep same ratio of wins/losses in both sets
    )
    
    print(f"Training set: {len(X_train)} games")
    print(f"Testing set: {len(X_test)} games")
    
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model"""
    
    print("\n=== Training Logistic Regression ===")
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    print("✓ Training complete")
    
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    
    print("\n=== Training Random Forest ===")
    
    model = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=10,      # Depth of each tree
        random_state=42
    )
    model.fit(X_train, y_train)
    
    print("✓ Training complete")
    
    return model


def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting model"""
    
    print("\n=== Training Gradient Boosting ===")
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    print("✓ Training complete")
    
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """
    Test model accuracy and show detailed metrics
    """
    
    print(f"\n=== Evaluating {model_name} ===")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Show detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"True Negatives (Away wins predicted correctly): {cm[0][0]}")
    print(f"False Positives (Away wins predicted as Home): {cm[0][1]}")
    print(f"False Negatives (Home wins predicted as Away): {cm[1][0]}")
    print(f"True Positives (Home wins predicted correctly): {cm[1][1]}")
    
    return accuracy


def show_feature_importance(model, feature_columns):
    """
    For tree-based models, show which features matter most
    """
    
    if hasattr(model, 'feature_importances_'):
        print("\n=== Feature Importance ===")
        
        # Get importances
        importances = model.feature_importances_
        
        # Create DataFrame for sorting
        feature_importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': importances
        })
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
        
        print(feature_importance_df)
    else:
        print(f"{model.__class__.__name__} doesn't have feature importances")


def save_model(model, filepath):
    """Save trained model to file"""
    
    joblib.dump(model, filepath)
    print(f"\n✓ Model saved to {filepath}")


def main():
    """Main training pipeline"""
    
    # Step 1: Load data
    df = load_processed_data('../data/processed_games.csv')
    
    # Step 2: Prepare features
    X, y, feature_columns = prepare_features(df)
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 4: Train all models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)
    
    # Step 5: Evaluate all models
    lr_accuracy = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_accuracy = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    gb_accuracy = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
    
    # Step 6: Show feature importance for best tree model
    print("\n" + "="*50)
    if rf_accuracy >= gb_accuracy:
        show_feature_importance(rf_model, feature_columns)
        best_model = rf_model
        best_name = "Random Forest"
        best_accuracy = rf_accuracy
    else:
        show_feature_importance(gb_model, feature_columns)
        best_model = gb_model
        best_name = "Gradient Boosting"
        best_accuracy = gb_accuracy
    
    # Step 7: Compare all models
    print("\n" + "="*50)
    print("=== MODEL COMPARISON ===")
    print(f"Logistic Regression: {lr_accuracy * 100:.2f}%")
    print(f"Random Forest: {rf_accuracy * 100:.2f}%")
    print(f"Gradient Boosting: {gb_accuracy * 100:.2f}%")
    print(f"\nBest Model: {best_name} ({best_accuracy * 100:.2f}%)")
    
    # Step 8: Save best model
    save_model(best_model, '../data/nba_model.pkl')
    
    print("\n" + "="*50)
    print("✓ Training complete!")


if __name__ == "__main__":
    main()