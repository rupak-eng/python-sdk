import httpx2
import pytest
from httpx2 import ASGITransport
from pydantic import AnyHttpUrl
from starlette.applications import Starlette

from mcp.server.auth.routes import build_metadata, create_auth_routes, validate_issuer_url
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
from tests.server.mcpserver.auth.test_auth_integration import MockOAuthProvider


def test_validate_issuer_url_https_allowed():
    validate_issuer_url(AnyHttpUrl("https://example.com/path"))


def test_validate_issuer_url_http_localhost_allowed():
    validate_issuer_url(AnyHttpUrl("http://localhost:8080/path"))


def test_validate_issuer_url_http_127_0_0_1_allowed():
    validate_issuer_url(AnyHttpUrl("http://127.0.0.1:8080/path"))


def test_validate_issuer_url_http_ipv6_loopback_allowed():
    validate_issuer_url(AnyHttpUrl("http://[::1]:8080/path"))


def test_validate_issuer_url_http_non_loopback_rejected():
    with pytest.raises(ValueError, match="Issuer URL must be HTTPS"):
        validate_issuer_url(AnyHttpUrl("http://evil.com/path"))


def test_validate_issuer_url_http_127_prefix_domain_rejected():
    """A domain like 127.0.0.1.evil.com is not loopback."""
    with pytest.raises(ValueError, match="Issuer URL must be HTTPS"):
        validate_issuer_url(AnyHttpUrl("http://127.0.0.1.evil.com/path"))


def test_validate_issuer_url_http_127_prefix_subdomain_rejected():
    """A domain like 127.0.0.1something.example.com is not loopback."""
    with pytest.raises(ValueError, match="Issuer URL must be HTTPS"):
        validate_issuer_url(AnyHttpUrl("http://127.0.0.1something.example.com/path"))


def test_validate_issuer_url_fragment_rejected():
    with pytest.raises(ValueError, match="fragment"):
        validate_issuer_url(AnyHttpUrl("https://example.com/path#frag"))


def test_validate_issuer_url_query_rejected():
    with pytest.raises(ValueError, match="query"):
        validate_issuer_url(AnyHttpUrl("https://example.com/path?q=1"))


def test_auth_settings_preserves_path_less_issuer():
    """A path-less issuer passed as a string keeps its canonical form (no trailing slash)."""
    settings = AuthSettings(
        issuer_url="https://as.example.com",  # type: ignore[arg-type]
        resource_server_url="https://rs.example.com",  # type: ignore[arg-type]
        validate_token_resource=True,
    )
    assert str(settings.issuer_url) == "https://as.example.com"
    assert str(settings.resource_server_url) == "https://rs.example.com"


def test_build_metadata_serves_issuer_without_trailing_slash():
    """The served issuer matches the configured one exactly (RFC 8414/9207 string comparison)."""
    settings = AuthSettings(
        issuer_url="https://as.example.com",  # type: ignore[arg-type]
        resource_server_url="https://rs.example.com",  # type: ignore[arg-type]
        validate_token_resource=True,
    )
    metadata = build_metadata(settings.issuer_url, None, ClientRegistrationOptions(), RevocationOptions())

    served = metadata.model_dump(mode="json", exclude_none=True)
    assert served["issuer"] == "https://as.example.com"
    assert served["authorization_endpoint"] == "https://as.example.com/authorize"
    assert served["token_endpoint"] == "https://as.example.com/token"


# Non-preflight OPTIONS handling on the CORS-wrapped auth routes (issue #3652).

VALID_CLIENT_METADATA = {
    "redirect_uris": ["https://client.example.com/callback"],
    "token_endpoint_auth_method": "client_secret_post",
    "grant_types": ["authorization_code", "refresh_token"],
    "response_types": ["code"],
    "client_name": "Test Client",
}


@pytest.fixture
def oauth_provider() -> MockOAuthProvider:
    return MockOAuthProvider()


@pytest.fixture
def cors_app(oauth_provider: MockOAuthProvider) -> Starlette:
    auth_routes = create_auth_routes(
        oauth_provider,
        issuer_url=AnyHttpUrl("http://localhost"),
        client_registration_options=ClientRegistrationOptions(enabled=True),
        revocation_options=RevocationOptions(enabled=True),
    )
    return Starlette(routes=auth_routes)


@pytest.fixture
def cors_client(cors_app: Starlette) -> httpx2.AsyncClient:
    transport = ASGITransport(app=cors_app)
    return httpx2.AsyncClient(transport=transport, base_url="http://localhost")


@pytest.mark.anyio
async def test_options_register_without_cors_headers_answers_204_and_registers_no_client(
    cors_client: httpx2.AsyncClient, oauth_provider: MockOAuthProvider
):
    """Regression test for #3652: a plain OPTIONS request must be answered empty
    instead of falling through CORSMiddleware to the body-reading handler."""
    response = await cors_client.options("/register")

    assert response.status_code == 204
    assert oauth_provider.clients == {}


@pytest.mark.anyio
async def test_options_register_with_json_body_answers_204_and_registers_no_client(
    cors_client: httpx2.AsyncClient, oauth_provider: MockOAuthProvider
):
    """Regression test for #3652: an OPTIONS carrying a registration payload must
    not reach the handler, so no client credentials are minted."""
    response = await cors_client.request(
        "OPTIONS",
        "/register",
        content=b'{"redirect_uris": ["https://client.example/cb"], "client_name": "demo"}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 204
    assert oauth_provider.clients == {}


@pytest.mark.anyio
async def test_options_token_without_cors_headers_answers_204(cors_client: httpx2.AsyncClient):
    """Regression test for #3652: /token has the same _cors wrapping, so plain
    OPTIONS there is answered empty too."""
    response = await cors_client.options("/token")

    assert response.status_code == 204


@pytest.mark.anyio
async def test_options_with_origin_but_no_preflight_headers_answers_204(cors_client: httpx2.AsyncClient):
    """An Origin header alone is not a CORS preflight (Access-Control-Request-Method
    is also required), so it is answered empty like any other plain OPTIONS."""
    response = await cors_client.options("/register", headers={"Origin": "https://client.example"})

    assert response.status_code == 204
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.anyio
async def test_preflight_options_register_still_answered_by_cors_middleware(
    cors_client: httpx2.AsyncClient, oauth_provider: MockOAuthProvider
):
    """A true preflight keeps flowing into CORSMiddleware and gets the usual CORS
    headers; the short-circuit only handles non-preflight OPTIONS."""
    response = await cors_client.options(
        "/register",
        headers={
            "Origin": "https://client.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "POST" in response.headers["access-control-allow-methods"]
    assert oauth_provider.clients == {}


@pytest.mark.anyio
async def test_post_register_still_registers_client(cors_client: httpx2.AsyncClient, oauth_provider: MockOAuthProvider):
    """The normal registration flow is untouched: POST still mints and stores a client."""
    response = await cors_client.post("/register", json=VALID_CLIENT_METADATA)

    assert response.status_code == 201
    client_id = response.json()["client_id"]
    assert client_id in oauth_provider.clients
