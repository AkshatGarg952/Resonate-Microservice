"""
Rate limiter configuration.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

# Rate limiter — uses client IP as key
limiter = Limiter(key_func=get_remote_address)
