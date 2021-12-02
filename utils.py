def flush(step: int, end: int):
    """
    Flush the console (only esthetical purposes).
    We want to do this before reaching the end,
    otherwise, we will get a blank output at the end
    """
    return print("\r", end="") if step != (end - 1) else print()
