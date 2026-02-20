"""
Internal API authentication dependency.

All microservice endpoints are internal — they should only be callable
by the Resonate-Server backend, not by anyone on the internet.

How it works:
  - Server sends header: X-Internal-Secret: <INTERNAL_API_SECRET>
  - Microservice checks it matches the env var
  - Returns 403 if missing or wrong

Setup:
  - Add INTERNAL_API_SECRET=<random-long-string> to both .env files
  - Generate one with: python -c "import secrets; print(secrets.token_hex(32))"
"""
import os
from fastapi import Header, HTTPException, Depends
from typing import Annotated


SECRET = os.getenv("INTERNAL_API_SECRET", "")


def verify_internal_secret(x_internal_secret: Annotated[str, Header()] = "") -> None:
    """FastAPI dependency — validates the shared internal secret header."""
    if not SECRET:
        # If the env var is not set, block all requests to prevent accidental exposure
        raise HTTPException(
            status_code=503,
            detail="Service not configured (INTERNAL_API_SECRET not set)"
        )
    if x_internal_secret != SECRET:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: invalid or missing internal secret"
        )
