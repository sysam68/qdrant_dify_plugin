from __future__ import annotations

import logging
from typing import Any

import httpx

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

from utils.qdrant_helpers import build_headers, resolve_endpoint

logger = logging.getLogger(__name__)


class QdrantProvider(ToolProvider):
    """
    Qdrant plugin provider for Dify that validates credentials and manages connections
    to Qdrant vector database instances.
    """
    DEFAULT_TIMEOUT = 10.0

    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        """
        Validates Qdrant connection credentials by performing format checks and
        establishing a test connection to the Qdrant instance.
        
        This method performs the following validation steps:
        1. Validates that base_url is provided and not empty
        2. Builds request headers (including API key and extra headers if provided)
        3. Establishes a connection to the Qdrant instance
        4. Verifies the instance is responsive by calling the collections API
        5. Validates authentication if API key is provided
        
        Args:
            credentials (dict[str, Any]): Dictionary containing connection credentials
                with the following keys:
                - base_url (str): Qdrant instance URL (e.g., https://your-instance.com)
                - api_key (str, optional): API key for authentication if required
                - extra_headers (dict or str, optional): Additional HTTP headers (JSON format)
                
        Raises:
            ToolProviderCredentialValidationError: If any validation step fails:
                - Missing base_url
                - Invalid extra_headers format
                - Connection failure
                - Authentication failure (401)
                - Server error (400+)
                
        Note:
            The method uses a simple GET request to verify connectivity and authentication.
            Supports both URL formats:
            - With port: https://xxx.cloud.qdrant.io:6333 (recommended by official docs)
            - Without port: https://xxx.cloud.qdrant.io (also works, defaults to port 443)
            This is a lightweight operation that doesn't modify any data.
        """
        base_url = (credentials.get("base_url") or "").strip()
        if not base_url:
            raise ToolProviderCredentialValidationError(
                "`base_url` is required. Please provide a valid Qdrant instance URL.\n\n"
                "If you don't have a Qdrant instance yet:\n"
                "1. Create a free Qdrant Cloud account at https://cloud.qdrant.io/\n"
                "2. Create a cluster and get the endpoint URL\n"
                "3. Or deploy Qdrant locally using Docker: docker run -p 6333:6333 qdrant/qdrant\n\n"
                "For more details, see SETUP_GUIDE.md"
            )

        # Build headers (includes API key and extra headers validation)
        headers, header_error = build_headers(credentials, include_content_type=False)
        if header_error:
            raise ToolProviderCredentialValidationError(header_error)

        # Try multiple endpoints for validation (in order of preference)
        # Qdrant supports both /collections (without /v1/) and /v1/collections
        # Both work, but /collections is the standard data API endpoint
        # Also try root path / as a lightweight health check
        validation_endpoints = ["/collections", "/v1/collections", "/"]
        last_error = None
        
        for endpoint_path in validation_endpoints:
            endpoint = resolve_endpoint(base_url, endpoint_path)
            try:
                response = httpx.get(endpoint, headers=headers, timeout=self.DEFAULT_TIMEOUT)
                
                # 200-299: Success
                if 200 <= response.status_code < 300:
                    logger.debug(f"Successfully validated Qdrant connection to {base_url} via {endpoint_path}")
                    return
                
                # 401: Authentication failed
                if response.status_code == 401:
                    raise ToolProviderCredentialValidationError(
                        "Invalid API key for Qdrant. Please verify your API key is correct."
                    )
                
                # 403: Permission denied, but connection and authentication succeeded
                # For Qdrant Cloud, some API keys may not have read permissions but can write
                # We consider this as validation passed since connection and auth succeeded
                if response.status_code == 403:
                    logger.warning(
                        f"Qdrant returned 403 for {endpoint_path}. "
                        f"This may indicate insufficient permissions, but connection and authentication succeeded. "
                        f"Please verify API key permissions in Qdrant Cloud if operations fail."
                    )
                    # Consider validation passed (connection and authentication verified)
                    return
                
                # 404: Endpoint not found, try next endpoint
                if response.status_code == 404:
                    last_error = f"Endpoint {endpoint_path} not found"
                    continue
                
                # Other 4xx/5xx errors
                if response.status_code >= 400:
                    last_error = f"Qdrant returned error status {response.status_code}: {response.text[:200] if response.text else 'No error details'}"
                    continue
                    
            except httpx.ConnectError as exc:
                raise ToolProviderCredentialValidationError(
                    f"Failed to connect to Qdrant at {base_url}. "
                    f"Please verify the URL is correct and the instance is reachable. Details: {exc}"
                ) from exc
            except httpx.TimeoutException as exc:
                raise ToolProviderCredentialValidationError(
                    f"Connection to Qdrant at {base_url} timed out. "
                    f"Please check your network connection and try again. Details: {exc}"
                ) from exc
            except httpx.HTTPError as exc:
                raise ToolProviderCredentialValidationError(
                    f"Failed to reach Qdrant at {base_url}. Details: {exc}"
                ) from exc
        
        # If all endpoints failed, raise the last error
        if last_error:
            raise ToolProviderCredentialValidationError(
                f"Failed to validate Qdrant connection. {last_error}. "
                f"Please verify the URL is correct: {base_url}"
            )
        
        # Should not reach here, but just in case
        raise ToolProviderCredentialValidationError(
            f"Failed to validate Qdrant connection at {base_url}. Please verify the URL and API key."
        )

    #########################################################################################
    # If OAuth is supported, uncomment the following functions.
    # Warning: please make sure that the sdk version is 0.4.2 or higher.
    #########################################################################################
    # def _oauth_get_authorization_url(self, redirect_uri: str, system_credentials: Mapping[str, Any]) -> str:
    #     """
    #     Generate the authorization URL for qdrant OAuth.
    #     """
    #     try:
    #         """
    #         IMPLEMENT YOUR AUTHORIZATION URL GENERATION HERE
    #         """
    #     except Exception as e:
    #         raise ToolProviderOAuthError(str(e))
    #     return ""
        
    # def _oauth_get_credentials(
    #     self, redirect_uri: str, system_credentials: Mapping[str, Any], request: Request
    # ) -> Mapping[str, Any]:
    #     """
    #     Exchange code for access_token.
    #     """
    #     try:
    #         """
    #         IMPLEMENT YOUR CREDENTIALS EXCHANGE HERE
    #         """
    #     except Exception as e:
    #         raise ToolProviderOAuthError(str(e))
    #     return dict()

    # def _oauth_refresh_credentials(
    #     self, redirect_uri: str, system_credentials: Mapping[str, Any], credentials: Mapping[str, Any]
    # ) -> OAuthCredentials:
    #     """
    #     Refresh the credentials
    #     """
    #     return OAuthCredentials(credentials=credentials, expires_at=-1)
