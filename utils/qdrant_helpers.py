"""
Common utility functions for Qdrant plugin

These functions can be shared between tools and provider to avoid code duplication.
"""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import urljoin


def build_headers(credentials: dict[str, Any], include_content_type: bool = False) -> tuple[dict[str, str], str | None]:
    """
    Build Qdrant API request headers
    
    Args:
        credentials: Credentials dictionary containing api_key and extra_headers
        include_content_type: Whether to include Content-Type header (needed for tools, not for provider)
    
    Returns:
        Tuple of (headers_dict, error_message), error_message is None if successful
    """
    import logging
    logger = logging.getLogger(__name__)
    
    headers: dict[str, str] = {}
    
    if include_content_type:
        headers["Content-Type"] = "application/json"
    
    # Check api_key field (support multiple possible field names for compatibility with different Dify versions)
    api_key = (
        credentials.get("api_key") or 
        credentials.get("api-key") or 
        credentials.get("API_KEY") or
        credentials.get("apiKey") or  # camelCase
        credentials.get("apikey")      # lowercase
    )
    
    # Debug: Log credentials keys and whether api_key exists
    logger.debug(f"build_headers: credentials keys={list(credentials.keys())}, "
                f"api_key present={bool(api_key)}, "
                f"api_key type={type(api_key).__name__ if api_key else 'None'}")
    
    if api_key:
        # Ensure api_key is a string, strip whitespace
        api_key = str(api_key).strip()
        if api_key:
            # Qdrant Cloud uses api-key header (official recommended format)
            # Note: Authorization: apikey <key> format is not supported and will cause 403 errors
            # Authorization: Bearer <key> is also supported, but api-key is more reliable
            headers["api-key"] = api_key
            logger.debug(f"build_headers: API key added to headers (length={len(api_key)})")
        else:
            logger.warning("build_headers: api_key is empty after stripping")
    else:
        logger.debug("build_headers: No API key found in credentials")

    extra_headers_raw = credentials.get("extra_headers")
    if extra_headers_raw:
        if isinstance(extra_headers_raw, dict):
            extra_headers = extra_headers_raw
        else:
            try:
                extra_headers = json.loads(extra_headers_raw)
            except json.JSONDecodeError as exc:
                return {}, f"Invalid extra_headers JSON: {exc}"
        if not isinstance(extra_headers, dict):
            return {}, "extra_headers must be a JSON object"
        headers.update({str(k): str(v) for k, v in extra_headers.items()})

    return headers, None


def resolve_endpoint(base_url: str, path_or_url: str, use_cloud_api: bool = False) -> str:
    """
    Resolve Qdrant API endpoint URL
    
    According to Qdrant Cloud official documentation:
    - All operations (data operations and collection management) use Database API (:6333 port)
    - Database API Key is used for all database operations, including:
      - Data operations: upsert, query, search, scroll, delete points
      - Collection management: create_collection, get_collection_info, delete_collection
    
    Args:
        base_url: Qdrant base URL (recommended format: https://xxx.cloud.qdrant.io:6333)
        path_or_url: Path or complete URL
        use_cloud_api: Deprecated, kept for backward compatibility. All operations use :6333 port
    
    Returns:
        Complete endpoint URL
    """
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    
    # Note: According to official docs, all operations are on :6333 port
    # use_cloud_api parameter is deprecated but kept for backward compatibility
    # If base_url has no port, default to :6333
    if ":" not in base_url.split("//")[1] if "//" in base_url else base_url:
        # If no port, add :6333
        if base_url.endswith("/"):
            base_url = base_url.rstrip("/")
        if not base_url.endswith(":6333"):
            base_url = f"{base_url}:6333"
    
    base = base_url.rstrip("/") + "/"
    relative = path_or_url.lstrip("/")
    return urljoin(base, relative)

