# ------------------------------------------------------------
#  quotex_feed/__init__.py  –  FIXED MOCK
# ------------------------------------------------------------
import pandas as pd
import numpy as np

# --------------------  ALL 100+ QUOTEX PAIRS  --------------------
ALL_QUOTEX = [
    # OTC FOREX
    "USD/BRL-OTC","USD/MXN-OTC","GBP/NZD-OTC","NZD/CHF-OTC","USD/INR-OTC","USD/ARS-OTC","USD/BDT-OTC","USD/COP-OTC","CAD/CHF-OTC",
    "USD/EGP-OTC","AUD/NZD-OTC","EUR/NZD-OTC","USD/PHP-OTC","NZD/USD-OTC","GBP/CAD","NZD/JPY-OTC","EUR/SGD-OTC","NZD/CAD-OTC",
    "USD/DZD-OTC","USD/IDR-OTC","USD/NGN-OTC","USD/PKR-OTC","USD/TRY-OTC",
    # REAL FOREX
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

# --------------------  FIXED MOCK FETCH_OHLC  --------------------
def fetch_ohlc(symbol, timeframe, candles=100):
    """
    Returns realistic mock OHLC data for any Quotex symbol.
    """
    # Base price dictionary (simplified for all symbols)
    base_prices = {
        'EURUSD': 1.08, 'GBPUSD': 1.27, 'USDJPY': 150, 'USDCHF': 0.88,
        'AUDUSD': 0.65, 'NZDUSD': 0.60, 'USDCAD': 1.36, 'BTCUSD': 45000,
        'ETHUSD': 3000, 'XAUUSD': 2000, 'US30': 38000, 'US100': 17000, 'US500': 5200,
        'USD/BRL': 4.95, 'USD/MXN': 17.0, 'USD/INR': 83.0, 'USD/ARS': 850,
        'EURUSD-OTC': 1.08, 'GBPUSD-OTC': 1.27,  # add OTC variants as needed
    }
    
    clean_symbol = symbol.replace('-OTC', '')
    base_price = base_prices.get(clean_symbol, 100)  # default fallback
    
    # Volatility per timeframe
    vol_map = {'1m': 0.001, '5m': 0.002, '15m': 0.003, '1h': 0.005}
    vol = vol_map.get(timeframe, 0.001)
    
    # Generate price series
    returns = np.random.normal(0, vol, candles)
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({'close': close_prices})
    df['open'] = df['close'].shift(1)
    df['open'].iloc[0] = base_price
    
    # Create high/low from open/close ONLY
    intraday_vol = vol * 1.5
    high_mult = 1 + np.random.uniform(0, intraday_vol, candles)
    low_mult = 1 - np.random.uniform(0, intraday_vol, candles)
    
    # Calculate temp high/low
    high_vals = df[['open', 'close']].max(axis=1) * high_mult
    low_vals = df[['open', 'close']].min(axis=1) * low_mult
    
    # Assign to df
    df['high'] = high_vals
    df['low'] = low_vals
    
    # Ensure ordering (high >= max, low <= min)
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df[['open', 'high', 'low', 'close']]
