import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

class MetaEnsemblePredictor:
    def __init__(self):
        # Base models with different biases
        self.models = {
            'rf': RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=42),
            'lr': LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        }
        # Meta-learner
        self.meta_model = LogisticRegression(random_state=42)
        self.is_trained = False
        
    def extract_features(self, candles):
        """Extract microstructure features"""
        features = []
        for i in range(len(candles) - 10):
            window = candles[i:i+10]
            features.append([
                np.mean([c['close'] - c['open'] for c in window]),  # Momentum
                np.std([c['high'] - c['low'] for c in window]),     # Volatility
                sum(1 for c in window if c['close'] > c['open']),   # Green count
                (window[-1]['volume'] / np.mean([c['volume'] for c in window[:-1]])) if window[-1]['volume'] else 1.0,  # Volume ratio
                (window[-1]['spread'] - np.mean([c['spread'] for c in window[:-1]])) / max(np.std([c['spread'] for c in window]), 0.0001),  # Spread pressure
                (window[-1]['close'] - window[-3]['close']) / max(window[-1]['atr'], 0.0001),  # Momentum exhaustion
            ])
        return np.array(features)
    
    def train(self, historical_candles):
        """Train on historical data"""
        X = self.extract_features(historical_candles)
        y = np.array([1 if c['close'] > c['open'] else 0 for c in historical_candles[10:]])
        
        # Train base models
        predictions = np.zeros((len(X), len(self.models)))
        for i, (name, model) in enumerate(self.models.items()):
            model.fit(X, y)
            predictions[:, i] = model.predict_proba(X)[:, 1]
        
        # Train meta-learner
        self.meta_model.fit(predictions, y)
        self.is_trained = True
    
    def predict_proba(self, recent_candles):
        """Predict probability of green candle"""
        if not self.is_trained or len(recent_candles) < 10:
            return 0.5
        
        features = self.extract_features(recent_candles[-10:])[-1].reshape(1, -1)
        
        # Get base predictions
        base_preds = np.zeros((1, len(self.models)))
        for i, (name, model) in enumerate(self.models.items()):
            base_preds[:, i] = model.predict_proba(features)[:, 1]
        
        # Meta-model combines them
        return self.meta_model.predict_proba(base_preds)[0, 1]
