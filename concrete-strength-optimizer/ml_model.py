import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt

class ConcreteStrengthPredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = ['temperature', 'humidity', 'curing_type', 'hours']
        self.metrics = {}
        self.feature_importance = {}
        
    def load_data(self, filepath):
        """Load dataset from the given path"""
        print(f"Loading dataset from: {filepath}")
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}")
        
        # Load the data
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        print("\n=== Preprocessing Data ===")
        
        # Check for missing values
        if df.isnull().sum().any():
            print("Missing values found. Filling with column means...")
            df = df.fillna(df.mean())
        
        # Encode curing_type (categorical variable)
        df['curing_type_encoded'] = self.label_encoder.fit_transform(df['curing_type'])
        
        # Display encoding mapping
        encoding_map = dict(zip(self.label_encoder.classes_, 
                               self.label_encoder.transform(self.label_encoder.classes_)))
        print(f"Curing type encoding: {encoding_map}")
        
        # Prepare features and target
        X = df[['temperature', 'humidity', 'curing_type_encoded', 'hours']].values
        y = df['strength'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Features shape: {X_scaled.shape}")
        print(f"Target shape: {y.shape}")
        
        # Basic statistics
        print(f"\nTarget variable stats:")
        print(f"  Min: {y.min():.2f} MPa")
        print(f"  Max: {y.max():.2f} MPa")
        print(f"  Mean: {y.mean():.2f} MPa")
        print(f"  Std: {y.std():.2f} MPa")
        
        return X_scaled, y, df
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train Random Forest model with hyperparameter tuning"""
        print("\n=== Training Model ===")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Testing samples: {X_test.shape[0]}")
        
        # Create model with optimized parameters
        self.model = RandomForestRegressor(
            n_estimators=200,        # Number of trees
            max_depth=15,             # Maximum depth of trees
            min_samples_split=5,      # Minimum samples to split a node
            min_samples_leaf=2,       # Minimum samples at leaf node
            random_state=random_state,
            n_jobs=-1                  # Use all CPU cores
        )
        
        # Train the model
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'train': {
                'r2': r2_score(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'mae': mean_absolute_error(y_train, y_pred_train)
            },
            'test': {
                'r2': r2_score(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test)
            }
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        self.metrics['cv_mean'] = cv_scores.mean()
        self.metrics['cv_std'] = cv_scores.std()
        
        # Feature importance
        self.feature_importance = dict(zip(
            self.feature_names, 
            self.model.feature_importances_
        ))
        
        # Print results
        print("\n=== Model Performance ===")
        print(f"Train R² Score: {self.metrics['train']['r2']:.4f}")
        print(f"Test R² Score:  {self.metrics['test']['r2']:.4f}")
        print(f"Test RMSE:      {self.metrics['test']['rmse']:.4f} MPa")
        print(f"Test MAE:       {self.metrics['test']['mae']:.4f} MPa")
        print(f"Cross-val R²:   {self.metrics['cv_mean']:.4f} (±{self.metrics['cv_std']:.4f})")
        
        print("\n=== Feature Importance ===")
        for feature, importance in sorted(self.feature_importance.items(), 
                                         key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {importance:.4f} ({importance*100:.1f}%)")
        
        return self.model
    
    def predict(self, temperature, humidity, curing_type, hours):
        """Predict strength for given parameters"""
        # Encode curing type
        curing_encoded = self.label_encoder.transform([curing_type])[0]
        
        # Create feature array
        features = np.array([[temperature, humidity, curing_encoded, hours]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        strength = self.model.predict(features_scaled)[0]
        
        return strength
    
    def find_optimal_time(self, temperature, humidity, curing_type, target_strength,
                     min_hours=24, max_hours=720, step=24):  # Changed defaults!
      """Find minimum curing time to achieve target strength"""
      hours_range = range(min_hours, max_hours + 1, step)
      
      for hours in hours_range:
          strength = self.predict(temperature, humidity, curing_type, hours)
          if strength >= target_strength:
              return hours, strength
      
      return max_hours, self.predict(temperature, humidity, curing_type, max_hours)
    
    def batch_predict(self, test_cases):
        """Predict for multiple test cases"""
        results = []
        for temp, hum, method, hours in test_cases:
            strength = self.predict(temp, hum, method, hours)
            results.append({
                'temperature': temp,
                'humidity': hum,
                'curing_type': method,
                'hours': hours,
                'predicted_strength': round(strength, 2)
            })
        return pd.DataFrame(results)
    
    def compare_curing_methods(self, temperature, humidity, target_strength):
        """Compare all curing methods for given conditions"""
        methods = ['Air', 'Water', 'Steam']
        results = []
        
        # Cost rates (₹ per hour)
        cost_rates = {
            'Air': 1500,
            'Compound': 2000,   
            'Water': 2500,
            'Steam': 5000
        }
        
        for method in methods:
            optimal_hours, strength = self.find_optimal_time(
                temperature, humidity, method, target_strength
            )
            
            cost = optimal_hours * cost_rates[method]
            
            results.append({
                'Curing Method': method,
                'Optimal Hours': optimal_hours,
                'Predicted Strength (MPa)': round(strength, 2),
                'Target Achieved': '✅' if strength >= target_strength else '❌',
                'Cost (₹)': f"₹{cost:,.0f}",
                'Rate (₹/hr)': f"₹{cost_rates[method]:,}"
            })
        
        return pd.DataFrame(results)
    
    def save_model(self, filepath='concrete_model.pkl'):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        print(f"\n✅ Model saved to {filepath}")
    
    def load_model(self, filepath='concrete_model.pkl'):
        """Load a trained model"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.metrics = model_data['metrics']
            self.feature_importance = model_data['feature_importance']
            print(f"\n✅ Model loaded from {filepath}")
            return True
        return False
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        features = list(self.feature_importance.keys())
        importance = list(self.feature_importance.values())
        
        # Sort by importance
        sorted_idx = np.argsort(importance)
        features = [features[i] for i in sorted_idx]
        importance = [importance[i] for i in sorted_idx]
        
        # Create horizontal bar chart
        ax.barh(features, importance, color='#3D52A0')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_predictions_vs_actual(self, X_test, y_test):
        """Plot predictions vs actual values"""
        y_pred = self.model.predict(X_test)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.6, color='#3D52A0', edgecolors='white', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('Actual Strength (MPa)', fontsize=12)
        ax.set_ylabel('Predicted Strength (MPa)', fontsize=12)
        ax.set_title('Predictions vs Actual Values', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add R² value
        r2 = r2_score(y_test, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig

# Main execution
if __name__ == "__main__":
    # Dataset path
    dataset_path = r"E:\hackathons\SEM-4\CreaTech Hackk\concrete-strength-optimizer\dataset.csv"
    
    try:
        # Initialize predictor
        predictor = ConcreteStrengthPredictor()
        
        # Step 1: Load data
        df = predictor.load_data(dataset_path)
        
        # Step 2: Preprocess data
        X, y, df_processed = predictor.preprocess_data(df)
        
        # Step 3: Train model
        predictor.train_model(X, y)
        
        # Step 4: Save model
        predictor.save_model('concrete_strength_model.pkl')
        
        # Step 5: Test predictions
        print("\n=== Test Predictions ===")
        test_cases = [
            (30, 65, 'Steam', 24),
            (25, 70, 'Water', 48),
            (35, 50, 'Air', 36),      # Changed to 'Air'
            (28, 60, 'Steam', 12),
            (32, 55, 'Water', 72)
        ]
        
        results_df = predictor.batch_predict(test_cases)
        print(results_df.to_string(index=False))
        
        # Step 6: Compare curing methods
        print("\n=== Curing Method Comparison (Temp=28°C, Humidity=60%, Target=30 MPa) ===")
        
        # Override the compare_curing_methods for this test
        methods = ['Air', 'Compound', 'Water', 'Steam']
        results = []
        cost_rates = {'Air': 1500, 'Compound': 2000, 'Water': 2500, 'Steam': 5000}
        
        for method in methods:
            optimal_hours, strength = predictor.find_optimal_time(28, 60, method, 30)
            cost = optimal_hours * cost_rates[method]
            results.append({
                'Curing Method': method,
                'Optimal Hours': optimal_hours,
                'Strength (MPa)': round(strength, 2),
                'Target Met': '✅' if strength >= 30 else '❌',
                'Cost (₹)': f"₹{cost:,.0f}"
            })
        
        comparison_df = pd.DataFrame(results)
        print(comparison_df.to_string(index=False))
        
        # Step 7: Find optimal time
        # Step 7: Find optimal time
        print("\n=== Optimal Time Finder ===")
        temp, hum, method, target = 30, 65, 'Steam', 35
        hours, strength = predictor.find_optimal_time(temp, hum, method, target, 
                                                    min_hours=24, max_hours=720, step=24)
        print(f"Conditions: Temp={temp}°C, Humidity={hum}%, Method={method}, Target={target} MPa")
        print(f"Optimal time: {hours} hours ({hours/24:.1f} days)")
        print(f"Predicted strength: {strength:.2f} MPa")
        
        print("\n✅ ML Model training complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()