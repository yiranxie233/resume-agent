from __future__ import annotations

import json
import urllib.error
import unittest
from unittest.mock import patch

from app.core.model_gateway import (
    ChatMessage,
    CredentialStore,
    CredentialUnavailable,
    GatewayError,
    HttpResponse,
    ModelGateway,
    ModelProfile,
    ModelRole,
    OllamaAdapter,
    OpenAICompatibleAdapter,
    ProbeStatus,
    Provider,
    UrllibTransport,
    canonicalize_openai_base_url,
    coerce_model_profile,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 1000.0

    def __call__(self) -> float:
        return self.value


class FakeTransport:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def request(self, method, url, *, headers=None, json_body=None, timeout=60.0):
        self.calls.append((method, url, dict(headers or {}), json_body))
        route = self.routes.get((method.upper(), url))
        if route is None:
            return HttpResponse(404, {}, b'{"error":"not found"}')
        if isinstance(route, Exception):
            raise route
        status, payload = route
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return HttpResponse(status, {}, body)


class ModelGatewayTests(unittest.TestCase):
    def test_credential_store_is_scoped_and_expires(self):
        clock = FakeClock()
        store = CredentialStore(clock=clock)
        handle = store.put("top-secret", scope="model:p", ttl_seconds=5)
        self.assertEqual(store.get(handle.handle_id, scope="model:p"), "top-secret")
        with self.assertRaises(CredentialUnavailable):
            store.get(handle.handle_id, scope="model:other")
        clock.value += 6
        with self.assertRaises(CredentialUnavailable):
            store.get(handle.handle_id, scope="model:p")

    def test_profile_defaults_and_public_dict_redact_metadata(self):
        chat = ModelProfile.default_chat()
        embedding = ModelProfile.default_embedding()
        self.assertEqual(chat.model_name, "qwen2.5:7b")
        self.assertEqual(chat.generation.timeout_seconds, 120.0)
        self.assertEqual(chat.generation.max_retries, 1)
        self.assertEqual(embedding.model_name, "bge-m3")
        profile = chat.snapshot(version="v2", metadata={"api_key": "secret", "nested": {"token": "x"}})
        public = profile.public_dict()
        self.assertEqual(public["metadata"]["api_key"], "[redacted]")
        self.assertEqual(public["metadata"]["nested"]["token"], "[redacted]")
        self.assertNotIn("secret", json.dumps(public))

    def test_pydantic_like_profile_is_coerced_without_importing_schema(self):
        class Incoming:
            profile_id = "incoming"
            role = "chat"
            provider = "ollama"
            base_url = "http://127.0.0.1:11434"
            model_name = "qwen2.5:7b"
            config_version = 3
            credential_required = False
            context_window_tokens = 4096
            generation_params = {"max_output_tokens": 256}
            max_input_tokens = None
            capabilities = {}

        profile = coerce_model_profile(Incoming())
        self.assertEqual(profile.profile_version, "v3")
        self.assertEqual(profile.generation.max_output_tokens, 256)

    def test_profile_generation_timeout_and_retry_overrides_are_honoured(self):
        profile = coerce_model_profile(
            {
                "profile_id": "custom-timeout",
                "role": "chat",
                "provider": "openai_compatible",
                "base_url": "https://example.invalid/v1",
                "model_name": "test-model",
                "generation_params": {
                    "timeout_seconds": 75,
                    "max_retries": 0,
                    "retry_interval_seconds": 0.25,
                },
            }
        )

        self.assertEqual(profile.generation.timeout_seconds, 75)
        self.assertEqual(profile.generation.max_retries, 0)
        self.assertEqual(profile.generation.retry_interval_seconds, 0.25)

    def test_ollama_chat_probe_and_call(self):
        profile = ModelProfile.default_chat()
        base = profile.base_url
        transport = FakeTransport(
            {
                ("GET", base + "/api/version"): (200, {"version": "0.5"}),
                ("GET", base + "/api/tags"): (200, {"models": [{"name": profile.model_name}]}),
                ("POST", base + "/api/show"): (200, {"digest": "sha256:test"}),
                ("POST", base + "/api/chat"): (
                    200,
                    {"message": {"content": '{"ok":true}'}, "done": True},
                ),
            }
        )
        result = OllamaAdapter(transport).probe(profile)
        self.assertEqual(result.status, ProbeStatus.READY)
        self.assertTrue(result.capabilities["structured_json"])
        called = ModelGateway(transport=transport).chat(
            profile, [ChatMessage("user", "hello")], request_key="r1"
        )
        self.assertEqual(called.content, '{"ok":true}')
        self.assertEqual(called.request_key, "r1")

    def test_ollama_missing_model_is_distinct(self):
        profile = ModelProfile.default_embedding()
        base = profile.base_url
        transport = FakeTransport(
            {
                ("GET", base + "/api/version"): (200, {"version": "0.5"}),
                ("GET", base + "/api/tags"): (200, {"models": [{"name": "other:latest"}]}),
            }
        )
        result = OllamaAdapter(transport).probe(profile)
        self.assertEqual(result.status, ProbeStatus.MODEL_NOT_INSTALLED)
        self.assertEqual(result.error_code, "model_not_installed")

    def test_ollama_implicit_latest_tag_is_installed(self):
        profile = ModelProfile.default_embedding()
        base = profile.base_url
        transport = FakeTransport(
            {
                ("GET", base + "/api/version"): (200, {"version": "0.5"}),
                ("GET", base + "/api/tags"): (
                    200,
                    {"models": [{"name": profile.model_name + ":latest", "digest": "sha256:test"}]},
                ),
                ("POST", base + "/api/show"): (200, {"digest": "sha256:test"}),
                ("POST", base + "/api/embed"): (
                    200,
                    {"embeddings": [[3.0, 4.0], [3.0, 4.0]]},
                ),
            }
        )

        result = OllamaAdapter(transport).probe(profile)

        self.assertEqual(result.status, ProbeStatus.READY)
        self.assertEqual(result.model_digest, "sha256:test")

    def test_ollama_scan_returns_exact_inventory(self):
        base = "http://127.0.0.1:11434"
        transport = FakeTransport(
            {
                ("GET", base + "/api/version"): (200, {"version": "0.5"}),
                ("GET", base + "/api/tags"): (200, {"models": [{"name": "qwen2.5:7b"}]}),
            }
        )
        result = OllamaAdapter(transport).scan(base)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["installed_names"], ["qwen2.5:7b"])

    def test_ollama_unreachable_is_a_distinct_probe_state(self):
        from app.core.model_gateway import GatewayError

        class DownTransport:
            def request(self, *args, **kwargs):
                raise GatewayError("service_unreachable", "connection refused", retryable=True)

        result = OllamaAdapter(DownTransport()).probe(ModelProfile.default_chat())
        self.assertEqual(result.status, ProbeStatus.SERVICE_UNREACHABLE)
        self.assertEqual(result.error_code, "service_unreachable")
        self.assertTrue(result.retryable)

    def test_ollama_embedding_probe_and_normalization(self):
        profile = ModelProfile.default_embedding()
        base = profile.base_url
        transport = FakeTransport(
            {
                ("GET", base + "/api/version"): (200, {"version": "0.5"}),
                ("GET", base + "/api/tags"): (200, {"models": [{"name": profile.model_name}]}),
                ("POST", base + "/api/show"): (200, {"digest": "sha256:test"}),
                ("POST", base + "/api/embed"): (
                    200,
                    {"embeddings": [[3.0, 4.0], [3.0, 4.0]]},
                ),
            }
        )
        result = OllamaAdapter(transport).probe(profile)
        self.assertEqual(result.status, ProbeStatus.READY)
        self.assertEqual(result.dimension, 2)
        embedded = OllamaAdapter(transport).embed(profile, ["a"])
        self.assertEqual(embedded.dimension, 2)
        self.assertAlmostEqual(sum(x * x for x in embedded.vectors[0]), 1.0)

    def test_chat_and_embedding_ports_reject_wrong_roles(self):
        gateway = ModelGateway()
        with self.assertRaises(ValueError):
            gateway.chat(ModelProfile.default_embedding(), [ChatMessage("user", "x")])
        with self.assertRaises(ValueError):
            gateway.embed(ModelProfile.default_chat(), ["x"])

    def test_openai_endpoint_normalization_and_credential_header(self):
        profile = ModelProfile(
            profile_id="openai-chat",
            role=ModelRole.CHAT,
            provider=Provider.OPENAI_COMPATIBLE,
            base_url="https://example.invalid/v1/",
            model_name="gpt-test",
            credential_required=True,
        )
        base = "https://example.invalid/v1"
        transport = FakeTransport(
            {
                ("POST", base + "/chat/completions"): (
                    200,
                    {"choices": [{"message": {"content": '{"ok":true}'}}]},
                )
            }
        )
        store = CredentialStore()
        handle = store.put("key", scope="model:openai-chat", ttl_seconds=300)
        result = ModelGateway(transport=transport, credential_store=store).chat(
            profile,
            [ChatMessage("user", "test")],
            credential_handle_id=handle.handle_id,
        )
        self.assertEqual(result.content, '{"ok":true}')
        self.assertEqual(transport.calls[0][1], base + "/chat/completions")
        self.assertEqual(transport.calls[0][2]["Authorization"], "Bearer key")

    def test_openai_raw_credential_header_for_compatible_relay(self):
        profile = ModelProfile(
            profile_id="relay-chat",
            role=ModelRole.CHAT,
            provider=Provider.OPENAI_COMPATIBLE,
            base_url="https://relay.example.invalid/v1",
            model_name="gpt-test",
            credential_required=True,
            auth_scheme="raw",
        )
        endpoint = profile.base_url + "/chat/completions"
        transport = FakeTransport(
            {
                ("POST", endpoint): (
                    200,
                    {"choices": [{"message": {"content": '{"ok":true}'}}]},
                )
            }
        )

        result = OpenAICompatibleAdapter(transport).chat(
            profile,
            [ChatMessage("user", "test")],
            secret="sk-private-test-value",
        )

        self.assertEqual(result.content, '{"ok":true}')
        self.assertEqual(
            transport.calls[0][2]["Authorization"],
            "sk-private-test-value",
        )
        self.assertNotIn("sk-private-test-value", json.dumps(profile.public_dict()))

    def test_openai_text_content_blocks_are_joined_as_text(self):
        profile = ModelProfile(
            profile_id="relay-blocks",
            role=ModelRole.CHAT,
            provider=Provider.OPENAI_COMPATIBLE,
            base_url="https://relay.example.invalid/v1",
            model_name="gpt-test",
        )
        endpoint = profile.base_url + "/chat/completions"
        transport = FakeTransport(
            {
                ("POST", endpoint): (
                    200,
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": [
                                        {"type": "text", "text": '{"ok":'},
                                        {"type": "text", "text": "true}"},
                                    ]
                                }
                            }
                        ]
                    },
                )
            }
        )

        result = OpenAICompatibleAdapter(transport).chat(
            profile, [ChatMessage("user", "return JSON")]
        )

        self.assertEqual(result.content, '{"ok":true}')

    def test_openai_compatible_reasoning_and_wrapped_output_shapes_are_supported(self):
        profile = ModelProfile(
            profile_id="relay-response-shapes",
            role=ModelRole.CHAT,
            provider=Provider.OPENAI_COMPATIBLE,
            base_url="https://relay.example.invalid/v1",
            model_name="reasoning-test",
        )
        endpoint = profile.base_url + "/chat/completions"
        for payload, expected in (
            (
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": None,
                                "reasoning_content": '{"sections":[]}',
                            },
                        }
                    ]
                },
                '{"sections":[]}',
            ),
            (
                {
                    "data": {
                        "choices": [
                            {"delta": {"content": '{"ok":true}'}}
                        ]
                    }
                },
                '{"ok":true}',
            ),
            (
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": '{"value":1}'}
                            ],
                        }
                    ]
                },
                '{"value":1}',
            ),
            (
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "emit_json",
                                            "arguments": '{"candidates":[]}',
                                        }
                                    }
                                ],
                            }
                        }
                    ]
                },
                '{"candidates":[]}',
            ),
        ):
            with self.subTest(expected=expected):
                transport = FakeTransport({("POST", endpoint): (200, payload)})
                result = OpenAICompatibleAdapter(transport).chat(
                    profile,
                    [ChatMessage("user", "return JSON")],
                )
                self.assertEqual(result.content, expected)

    def test_empty_length_response_retries_old_1200_profile_with_larger_budget(self):
        class LengthThenContentTransport:
            def __init__(self):
                self.calls = []

            def request(self, method, url, *, headers=None, json_body=None, timeout=60.0):
                body = dict(json_body or {})
                self.calls.append(body)
                if len(self.calls) == 1:
                    return HttpResponse(
                        200,
                        {},
                        json.dumps(
                            {
                                "choices": [
                                    {
                                        "finish_reason": "length",
                                        "message": {
                                            "content": "",
                                            # A truncated hidden chain must not
                                            # be treated as the final JSON.
                                            "reasoning_content": "partial reasoning",
                                        },
                                    }
                                ]
                            }
                        ).encode(),
                    )
                return HttpResponse(
                    200,
                    {},
                    b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"sections\\":[]}"}}]}',
                )

        profile = ModelProfile(
            profile_id="legacy-1200-profile",
            role=ModelRole.CHAT,
            provider=Provider.OPENAI_COMPATIBLE,
            base_url="https://relay.example.invalid/v1",
            model_name="reasoning-test",
            context_window_tokens=32768,
        )
        transport = LengthThenContentTransport()

        result = OpenAICompatibleAdapter(transport).chat(
            profile,
            [ChatMessage("user", "structure this resume")],
            response_format={"type": "json_object"},
        )

        self.assertEqual(result.content, '{"sections":[]}')
        self.assertEqual(transport.calls[0]["max_tokens"], 1200)
        self.assertEqual(transport.calls[1]["max_tokens"], 8192)
        self.assertNotIn("response_format", transport.calls[1])

    def test_nonempty_truncated_structured_response_is_retried_atomically(self):
        class TruncatedThenCompleteTransport:
            def __init__(self):
                self.calls = []

            def request(self, method, url, *, headers=None, json_body=None, timeout=60.0):
                body = dict(json_body or {})
                self.calls.append(body)
                if len(self.calls) == 1:
                    return HttpResponse(
                        200,
                        {},
                        b'{"choices":[{"finish_reason":"length","message":{"content":"{\\"responsibilities\\":[\\"partial"}}]}',
                    )
                return HttpResponse(
                    200,
                    {},
                    b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"responsibilities\\":[],\\"requirements\\":[],\\"skills\\":[]}"}}]}',
                )

        profile = ModelProfile(
            profile_id="legacy-truncated-profile",
            role=ModelRole.CHAT,
            provider=Provider.OPENAI_COMPATIBLE,
            base_url="https://relay.example.invalid/v1",
            model_name="reasoning-test",
            context_window_tokens=32768,
        )
        transport = TruncatedThenCompleteTransport()

        result = OpenAICompatibleAdapter(transport).chat(
            profile,
            [ChatMessage("user", "structure this long JD")],
            response_format={"type": "json_object"},
        )

        self.assertIn('"requirements":[]', result.content)
        self.assertEqual(transport.calls[0]["max_tokens"], 1200)
        self.assertEqual(transport.calls[1]["max_tokens"], 8192)
        self.assertNotIn("response_format", transport.calls[1])

    def test_empty_non_length_response_is_not_blindly_retried(self):
        profile = ModelProfile(
            profile_id="empty-stop-profile",
            role=ModelRole.CHAT,
            provider=Provider.OPENAI_COMPATIBLE,
            base_url="https://relay.example.invalid/v1",
            model_name="gpt-test",
        )
        endpoint = profile.base_url + "/chat/completions"
        transport = FakeTransport(
            {
                ("POST", endpoint): (
                    200,
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": ""},
                            }
                        ]
                    },
                )
            }
        )

        with self.assertRaises(GatewayError) as raised:
            OpenAICompatibleAdapter(transport).chat(
                profile,
                [ChatMessage("user", "return JSON")],
            )

        self.assertEqual(raised.exception.code, "invalid_response")
        self.assertEqual(len(transport.calls), 1)

    def test_openai_probe_uses_minimal_chat_completions_payload(self):
        profile = ModelProfile(
            profile_id="relay-probe",
            role=ModelRole.CHAT,
            provider=Provider.OPENAI_COMPATIBLE,
            base_url="https://relay.example.invalid/v1",
            model_name="gpt-test",
            credential_required=True,
            auth_scheme="raw",
        )
        endpoint = profile.base_url + "/chat/completions"
        transport = FakeTransport(
            {
                ("POST", endpoint): (
                    200,
                    {"choices": [{"message": {"content": '{"ok":true}'}}]},
                )
            }
        )

        result = OpenAICompatibleAdapter(transport).probe(
            profile, secret="sk-private-test-value"
        )

        self.assertEqual(result.status, ProbeStatus.READY)
        self.assertEqual(
            set(transport.calls[0][3]),
            {"model", "messages"},
        )
        self.assertNotIn("response_format", transport.calls[0][3])

    def test_openai_structured_response_retries_without_response_format(self):
        class ResponseFormatRejectingTransport:
            def __init__(self):
                self.calls = []

            def request(self, method, url, *, headers=None, json_body=None, timeout=60.0):
                self.calls.append((method, url, dict(headers or {}), dict(json_body or {})))
                if "response_format" in (json_body or {}):
                    return HttpResponse(
                        400,
                        {},
                        b'{"error":{"message":"response_format is not supported"}}',
                    )
                return HttpResponse(
                    200,
                    {},
                    b'{"choices":[{"message":{"content":"{\\"ok\\":true}"}}]}',
                )

        profile = ModelProfile(
            profile_id="relay-fallback",
            role=ModelRole.CHAT,
            provider=Provider.OPENAI_COMPATIBLE,
            base_url="https://relay.example.invalid/v1",
            model_name="gpt-test",
        )
        transport = ResponseFormatRejectingTransport()

        result = OpenAICompatibleAdapter(transport).chat(
            profile,
            [ChatMessage("user", "return JSON")],
            response_format={"type": "json_object"},
        )

        self.assertEqual(result.content, '{"ok":true}')
        self.assertEqual(len(transport.calls), 2)
        self.assertIn("response_format", transport.calls[0][3])
        self.assertNotIn("response_format", transport.calls[1][3])

    def test_auth_scheme_can_be_loaded_from_persisted_generation_params(self):
        profile = coerce_model_profile(
            {
                "profile_id": "persisted-relay",
                "role": "chat",
                "provider": "openai_compatible",
                "base_url": "https://relay.example.invalid/v1",
                "model_name": "gpt-test",
                "generation_params": {"auth_scheme": "raw"},
            }
        )

        self.assertEqual(profile.auth_scheme, "raw")

    def test_openai_full_chat_endpoint_is_canonicalized_to_v1_root(self):
        self.assertEqual(
            canonicalize_openai_base_url(
                "https://api.example.com/v1/chat/completions"
            ),
            "https://api.example.com/v1",
        )
        self.assertEqual(
            canonicalize_openai_base_url("https://api.example.com"),
            "https://api.example.com/v1",
        )

    def test_transport_distinguishes_network_permission_denied(self):
        denied = urllib.error.URLError(
            PermissionError(10013, "access permissions denied")
        )
        with (
            patch("urllib.request.urlopen", side_effect=denied),
            self.assertRaises(Exception) as caught,
        ):
            UrllibTransport().request("GET", "https://api.example.com/v1/models")

        self.assertEqual(getattr(caught.exception, "code", None), "network_permission_denied")
        self.assertNotIn("api.example.com", str(caught.exception))

    def test_gateway_probe_returns_credential_missing_without_secret(self):
        profile = ModelProfile(
            profile_id="openai-missing",
            role=ModelRole.CHAT,
            provider=Provider.OPENAI_COMPATIBLE,
            base_url="https://example.invalid/v1",
            model_name="gpt-test",
            credential_required=True,
        )
        result = ModelGateway().probe(profile)
        self.assertEqual(result.status, ProbeStatus.CREDENTIAL_MISSING)
        self.assertEqual(result.error_code, "credential_missing")
        self.assertTrue(result.requires_user)

    def test_chat_refuses_blocked_context_before_http(self):
        from app.core.model_gateway import GatewayError

        class Blocked:
            blocked = True
            blocked_reason = "context_window_unknown"

        transport = FakeTransport({})
        with self.assertRaises(GatewayError) as raised:
            ModelGateway(transport=transport).chat(
                ModelProfile.default_chat(),
                [ChatMessage("user", "x")],
                context_snapshot=Blocked(),
            )
        self.assertEqual(raised.exception.code, "context_budget_blocked")
        self.assertEqual(transport.calls, [])

    def test_external_openai_http_is_rejected_but_local_http_is_allowed(self):
        with self.assertRaises(ValueError):
            ModelProfile(
                profile_id="bad",
                role=ModelRole.CHAT,
                provider=Provider.OPENAI_COMPATIBLE,
                base_url="http://api.example.invalid",
                model_name="x",
            )
        local = ModelProfile(
            profile_id="local",
            role=ModelRole.CHAT,
            provider=Provider.OPENAI_COMPATIBLE,
            base_url="http://127.0.0.1:9000/v1",
            model_name="x",
        )
        self.assertEqual(local.base_url, "http://127.0.0.1:9000/v1")


if __name__ == "__main__":
    unittest.main()
