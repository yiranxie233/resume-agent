from __future__ import annotations

import json
import unittest

from app.core.model_gateway import (
    ChatMessage,
    CredentialStore,
    CredentialUnavailable,
    HttpResponse,
    ModelGateway,
    ModelProfile,
    ModelRole,
    OllamaAdapter,
    OpenAICompatibleAdapter,
    ProbeStatus,
    Provider,
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
