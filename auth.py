# -*- coding: utf-8 -*-
"""認證層:AUTH_MODE 環境變數切換。

  AUTH_MODE=none       本機/單人模式:不驗證,user_id 固定 'local'(預設,零設定可跑)
  AUTH_MODE=supabase   生產模式:驗證 Supabase Auth 簽發的 JWT
      - 設 SUPABASE_JWT_SECRET(專案 Settings → API → JWT Secret)→ HS256 驗證
      - 或只設 SUPABASE_URL → 自動用 JWKS(新式非對稱簽章)驗證
      - SUPABASE_URL / SUPABASE_ANON_KEY 同時供前端初始化(經 /api/config 下發)

使用者本體(email、密碼)由 Supabase 託管,本系統只持有 user_id(JWT 的 sub)。
"""
import os
from functools import wraps

from flask import g, jsonify, request

AUTH_MODE = os.environ.get("AUTH_MODE", "none").strip().lower()
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "").strip()
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "").strip()

_jwks_client = None


def _get_jwks_client():
    global _jwks_client
    if _jwks_client is None:
        from jwt import PyJWKClient
        _jwks_client = PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json",
                                   cache_keys=True)
    return _jwks_client


def verify_token(token: str) -> str:
    """驗證 JWT,回傳 user_id(sub)。失敗拋例外。"""
    import jwt as pyjwt
    if SUPABASE_JWT_SECRET:
        payload = pyjwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"],
                               audience="authenticated")
    else:
        key = _get_jwks_client().get_signing_key_from_jwt(token)
        payload = pyjwt.decode(token, key.key,
                               algorithms=["ES256", "RS256"], audience="authenticated")
    sub = payload.get("sub")
    if not sub:
        raise ValueError("token 缺少 sub")
    return str(sub)


def require_user(f):
    """API decorator:解析使用者身分到 g.user_id;supabase 模式下未登入回 401。"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if AUTH_MODE != "supabase":
            g.user_id = "local"
            return f(*args, **kwargs)
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify({"error": "請先登入。"}), 401
        try:
            g.user_id = verify_token(header[7:])
        except Exception:
            return jsonify({"error": "登入已過期或無效,請重新登入。"}), 401
        return f(*args, **kwargs)
    return wrapper


def public_config() -> dict:
    """前端初始化所需的公開設定(anon key 本來就是公開金鑰)。"""
    return {
        "auth_mode": AUTH_MODE,
        "supabase_url": SUPABASE_URL if AUTH_MODE == "supabase" else "",
        "supabase_anon_key": SUPABASE_ANON_KEY if AUTH_MODE == "supabase" else "",
    }
