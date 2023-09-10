

def window_splitter(signal, window_size, overlap=0):
    """Splits the signal into windows
    Parameters
    ----------
    signal : nd-array or pandas DataFrame
        input signal
    window_size : int
        number of points of window size
    overlap : float
        percentage of overlap, value between 0 and 1 (exclusive)
        Default: 0
    Returns
    -------
    list
        list of signal windows
    """
    if not isinstance(window_size, int):
        raise SystemExit('window_size must be an integer.')
    step = int(round(window_size)) if overlap == 0 else int(round(window_size * (1 - overlap)))
    if step == 0:
        raise SystemExit('Invalid overlap. '
                         'Choose a lower overlap value.')
    if len(signal) % window_size == 0 and overlap == 0:
        return [signal[i:i + window_size] for i in range(0, len(signal), step)]
    else:
        return [signal[i:i + window_size] for i in range(0, len(signal) - window_size + 1, step)]