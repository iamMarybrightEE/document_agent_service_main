from typing import Any

import jwt
from fastapi import Header, HTTPException, status

from app.core.config import settings


def decode_bearer_token(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    # Development mode: bypass authentication
    if settings.auth_mode == "dev":
        return {
            "sub": "dev-user",
            "tenant_id": "default",
        }
    
    if not authorization or not authorization.startswith("Bearer "):
        if settings.auth_mode == "dev_trust_bearer":
            return {
                "sub": "dev-user",
                "tenant_id": "default",
            }
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    token = authorization.split(" ", maxsplit=1)[1]
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        if settings.auth_mode == "dev_trust_bearer":
            return {
                "sub": "dev-user",
                "tenant_id": "default",
                "token_preview": token[:12],
                "auth_mode": "dev_trust_bearer",
            }
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
    return payload
