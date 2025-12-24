import pytz
from datetime import datetime, timedelta

def get_bdt_time():
    """Get current Bangladesh Time (UTC+6)"""
    utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
    bdt = utc_now.astimezone(pytz.timezone('Asia/Dhaka'))
    return bdt

def format_bdt_time(dt):
    """Format time as 3:30:00 (24-hour format)"""
    return dt.strftime("%H:%M:%S")

def generate_future_times(start_time, count=100, interval_minutes=1):
    """Generate future times in BDT for next candles"""
    times = []
    current = start_time
    for i in range(count):
        current += timedelta(minutes=interval_minutes)
        times.append(current)
    return times

def format_signal(pair, signal_time, direction):
    """Format: BDT/USD | TIME: 3:30:00 || UP/Call"""
    direction_text = "UP/Call" if direction == "LONG" else "DOWN/Put"
    return f"{pair} | TIME: {format_bdt_time(signal_time)} || {direction_text}"
