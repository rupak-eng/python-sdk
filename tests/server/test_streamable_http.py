"""Tests for the legacy streamable-HTTP server transport (`mcp.server.streamable_http`)."""

import anyio
import httpx2
import pytest

from mcp.server.streamable_http import StreamableHTTPServerTransport


@pytest.mark.anyio
async def test_notification_post_with_terminated_session_still_answers_202() -> None:
    """Regression test for https://github.com/modelcontextprotocol/python-sdk/issues/3631.

    Posting a notification whose session read stream closed after connect (simulating a
    terminated session) must still answer 202: no duplicate HTTP response may be attempted
    and no ClosedResourceError may escape. Closing the private writer is the only way to
    express that teardown through the public API.
    """
    transport = StreamableHTTPServerTransport(mcp_session_id=None, is_json_response_enabled=True)
    async with transport.connect():
        assert transport._read_stream_writer is not None
        await transport._read_stream_writer.aclose()
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=transport.handle_request, raise_app_exceptions=True),
            base_url="http://testserver",
        ) as client:
            # On the unfixed transport this either hangs or the duplicate-response
            # AssertionError escapes the ASGI app.
            with anyio.fail_after(5):
                response = await client.post(
                    "/mcp",
                    headers={"accept": "application/json", "content-type": "application/json"},
                    json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
                )
            assert response.status_code == 202
