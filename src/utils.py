# Utility functions for StyleX, including timestamp generation 
import time

def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
