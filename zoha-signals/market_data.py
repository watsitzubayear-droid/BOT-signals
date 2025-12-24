import numpy as np
from datetime import datetime

class MarketDataSimulator:
    def __init__(self, market_type='real'):
        self.market_type = market_type  # 'real' or 'otc'
        self.base_price = 1.0850  # Base price for EUR/USD
        
    def generate_candles(self, count=1000):
        """Generate synthetic 1-minute candles"""
        candles = []
        current_price = self.base_price
        
        for i in range(count):
            if self.market_type == 'otc':
                # OTC: More manipulation patterns
                trend = np.random.choice(['up', 'down', 'sideways'], p=[0.3, 0.3, 0.4])
                volatility = np.random.uniform(0.0005, 0.002)
                
                # Fake breakout pattern
                if i % 20 == 0:
                    wick_size = np.random.uniform(0.001, 0.003)
                    open_price = current_price
                    close_price = current_price + np.random.uniform(-wick_size, wick_size)
                    high = max(open_price, close_price) + wick_size
                    low = min(open_price, close_price) - wick_size
                else:
                    if trend == 'up':
                        current_price += np.random.uniform(0.0001, 0.0005)
                    elif trend == 'down':
                        current_price -= np.random.uniform(0.0001, 0.0005)
                    
                    open_price = current_price
                    close_price = current_price + np.random.uniform(-volatility, volatility)
                    high = max(open_price, close_price) + np.random.uniform(0, volatility/2)
                    low = min(open_price, close_price) - np.random.uniform(0, volatility/2)
                
                # Simulate fake volume
                volume = np.random.randint(50, 150)
                spread = np.random.uniform(0.0002, 0.0008)
                
            else:
                # Real market: More organic movement
                volatility = np.random.uniform(0.0001, 0.0008)
                trend_factor = 0.0002
                
                # Random walk with slight trend persistence
                drift = np.random.normal(0, 0.0001)
                current_price += drift + np.random.choice([-trend_factor, trend_factor])
                
                open_price = current_price
                close_price = current_price + np.random.uniform(-volatility, volatility)
                high = max(open_price, close_price) + np.random.uniform(0, volatility/2)
                low = min(open_price, close_price) - np.random.uniform(0, volatility/2)
                
                volume = np.random.randint(100, 500)
                spread = np.random.uniform(0.0001, 0.0005)
            
            candles.append({
                'timestamp': datetime.now().timestamp() + i * 60,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume,
                'spread': spread,
                'atr': volatility * 2
            })
            
            current_price = close_price
        
        return candles
    
    def get_recent_candles(self, count=20):
        """Get last N candles for feature extraction"""
        return self.generate_candles(count)
