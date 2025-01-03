def get_last_trading_day(history):
    """Returns the most recent trading day's close and the previous close."""
    # Sort the history by date and get the last two trading days
    valid_days = history.sort_index().tail(2)
    if len(valid_days) < 2:
        return None, None  # Insufficient data
    return valid_days['Close'].iloc[-1], valid_days['Close'].iloc[-2]