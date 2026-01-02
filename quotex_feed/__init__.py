# ------------------------------------------------------------
#  quotex_feed/__init__.py  –  FULL MOCK FOR ALL QUOTEX PAIRS
# ------------------------------------------------------------
import pandas as pd
import numpy as np

# --------------------  COMPLETE QUOTEX PAIR LIST  --------------------
ALL_QUOTEX = [
    # OTC FOREX
    "USD/BRL-OTC","USD/MXN-OTC","GBP/NZD-OTC","NZD/CHF-OTC","USD/INR-OTC","USD/ARS-OTC","USD/BDT-OTC","USD/COP-OTC","CAD/CHF-OTC",
    "USD/EGP-OTC","AUD/NZD-OTC","EUR/NZD-OTC","USD/PHP-OTC","NZD/USD-OTC","GBP/CAD","NZD/JPY-OTC","EUR/SGD-OTC","NZD/CAD-OTC",
    "USD/DZD-OTC","USD/IDR-OTC","USD/NGN-OTC","USD/PKR-OTC","USD/TRY-OTC",
    # REAL FOREX (major/minor)
    "EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","NZDUSD","USDCAD","EURGBP","EURJPY","GBPJPY","AUDJPY","NZDJPY",
    "CHFJPY","CADJPY","GBPCHF","EURCHF","AUDCHF","EURCAD","GBPCAD","AUDCAD","EURAUD","GBPAUD","EURNZD","GBPNZD",
    "AUDNZD","USDSGD","AUDSGD","GBPSGD",
    # INDICES
    "US30","US100","US500","GER30","UK100","AUS200","JPN225","ESP35","FRA40","STOXX50","FTSE100","NASDAQ",
    # CRYPTO
    "BTCUSD","ETHUSD","BNBUSD","SOLUSD","ADAUSD","XRPUSD","DOGEUSD","MATICUSD","DOTUSD","AVAXUSD","ATOMUSD",
    "LINKUSD","UNIUSD","LTCUSD","BCHUSD","XLMUSD","VETUSD","FILUSD","TRXUSD","ETCUSD",
    # COMMODITIES
    "XAUUSD","XAGUSD","BRENT","WTI","COPPER","NGAS","PALLADIUM","PLATINUM"
]

# --------------------  MOCK FETCH_OHLC  --------------------
def fetch_ohlc(symbol, timeframe, candles=100):
    """
    Returns realistic mock OHLC data for any Quotex symbol.
    Base prices and volatility are tailored to each asset class.
    """
    # Base price lookup by asset class
    base_prices = {
        # OTC/Real Forex (major)
        'EURUSD': 1.08, 'GBPUSD': 1.27, 'USDJPY': 150, 'USDCHF': 0.88,
        'AUDUSD': 0.65, 'NZDUSD': 0.60, 'USDCAD': 1.36,
        # Crosses
        'EURGBP': 0.85, 'EURJPY': 162, 'GBPJPY': 190, 'AUDJPY': 98,
        'NZDJPY': 90, 'CHFJPY': 170, 'CADJPY': 110, 'GBPCHF': 1.14,
        'EURCHF': 0.95, 'AUDCHF': 0.57, 'EURCAD': 1.47, 'GBPCAD': 1.73,
        'AUDCAD': 0.88, 'EURAUD': 1.66, 'GBPAUD': 1.95, 'EURNZD': 1.80,
        'GBPNZD': 2.12, 'AUDNZD': 1.08, 'USDSGD': 1.34,
        # OTC variants (strip -OTC for lookup)
        'USD/BRL': 4.95, 'USD/MXN': 17.0, 'GBP/NZD': 2.12, 'NZD/CHF': 0.53,
        'USD/INR': 83.0, 'USD/ARS': 850, 'USD/BDT': 110, 'USD/COP': 4000,
        'CAD/CHF': 0.64, 'USD/EGP': 31, 'AUD/NZD': 1.08, 'EUR/NZD': 1.80,
        'USD/PHP': 56, 'USD/DZD': 135, 'USD/IDR': 15600, 'USD/NGN': 800,
        'USD/PKR': 280, 'USD/TRY': 30,
        # Indices
        'US30': 38000, 'US100': 17000, 'US500': 5200, 'GER30': 18000,
        'UK100': 7600, 'AUS200': 7500, 'JPN225': 39000, 'ESP35': 11000,
        'FRA40': 7800, 'STOXX50': 4500, 'FTSE100': 7600, 'NASDAQ': 17000,
        # Crypto
        'BTCUSD': 45000, 'ETHUSD': 3000, 'BNBUSD': 400, 'SOLUSD': 150,
        'ADAUSD': 0.50, 'XRPUSD': 0.60, 'DOGEUSD': 0.08, 'MATICUSD': 1.0,
        'DOTUSD': 8, 'AVAXUSD': 40, 'ATOMUSD': 10, 'LINKUSD': 15,
        'UNIUSD': 8, 'LTCUSD': 70, 'BCHUSD': 250, 'XLMUSD': 0.12,
        'VETUSD': 0.03, 'FILUSD': 5, 'TRXUSD': 0.10, 'ETCUSD': 25,
        # Commodities
        'XAUUSD': 2000, 'XAGUSD': 25, 'BRENT': 85, 'WTI': 80,
        'COPPER': 4.5, 'NGAS': 2.5, 'PALLADIUM': 1000, 'PLATINUM': 1000
    }
    
    # Handle -OTC suffix
    clean_symbol = symbol.replace('-OTC', '') if '-OTC' in symbol else symbol
    base_price = base_prices.get(clean_symbol, 100)
    
    # Timeframe-based volatility
    vol_map = {'1m': 0.001, '5m': 0.002, '15m': 0.003, '1h': 0.005}
    vol = vol_map.get(timeframe, 0.001)
    
    # Generate random walk
    returns = np.random.normal(0, vol, candles)
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    # Build OHLC
    df = pd.DataFrame({'close': close_prices})
    df['open'] = df['close'].shift(1)
    df['open'].iloc[0] = base_price
    
    # Intraday range
    intraday_vol = vol * 1.5
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, intraday_vol, candles))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, intraday_vol, candles))
    
    # Ensure extremes
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df[['open', 'high', 'low', 'close']]
