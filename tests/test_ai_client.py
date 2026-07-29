import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import ai_client as ai_client_module
from ai_client import (
    AI_MAX_TOKENS_AZURE,
    AI_MAX_TOKENS_DEEPSEEK,
    AI_MAX_TOKENS_OPENAI,
    UnifiedAIClient,
    get_provider_max_tokens,
)


class _RecordingResponsesClient:
    def __init__(self, output_text="translated", status="completed"):
        self.kwargs = None
        self.output_text = output_text
        self.status = status

    def create(self, **kwargs):
        self.kwargs = kwargs
        incomplete_details = (
            SimpleNamespace(reason="max_output_tokens")
            if self.status == "incomplete"
            else None
        )
        return SimpleNamespace(
            output_text=self.output_text,
            status=self.status,
            incomplete_details=incomplete_details,
        )


class _RecordingChatCompletionsClient:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="translated")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")]
        )


class AIClientTokenLimitTest(unittest.TestCase):
    def test_provider_specific_environment_budget_is_used_and_clamped(self):
        with mock.patch.dict(
            ai_client_module.os.environ,
            {
                "OPENAI_MAX_OUTPUT_TOKENS": "1234",
                "DEEPSEEK_MAX_OUTPUT_TOKENS": "999999",
            },
            clear=False,
        ):
            self.assertEqual(get_provider_max_tokens("openai"), 1234)
            self.assertEqual(get_provider_max_tokens("deepseek"), 8192)

    def test_invalid_provider_specific_environment_budget_is_rejected(self):
        with mock.patch.dict(
            ai_client_module.os.environ,
            {"GEMINI_MAX_OUTPUT_TOKENS": "0"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                get_provider_max_tokens("gemini")

    def test_azure_gpt_uses_65536_token_limit(self):
        responses = _RecordingResponsesClient()
        ai_client = object.__new__(UnifiedAIClient)
        ai_client.provider = "azure"
        ai_client.model = "gpt-5.4"
        ai_client.max_tokens = AI_MAX_TOKENS_AZURE
        ai_client.client = SimpleNamespace(responses=responses)

        result = ai_client.chat_completion(
            [{"role": "user", "content": "Translate this document."}],
            max_tokens=100_000,
        )

        self.assertEqual(result, "translated")
        self.assertEqual(AI_MAX_TOKENS_AZURE, 65_536)
        self.assertEqual(responses.kwargs["max_output_tokens"], 65_536)
        self.assertNotEqual(
            responses.kwargs["max_output_tokens"],
            AI_MAX_TOKENS_DEEPSEEK,
        )
        self.assertEqual(result.completion_status, "complete")
        self.assertEqual(result.completion_reason, "")

    def test_provider_limits_do_not_fall_back_to_deepseek(self):
        self.assertEqual(get_provider_max_tokens("azure"), 65_536)
        with self.assertRaisesRegex(ValueError, "Unsupported AI provider"):
            get_provider_max_tokens("gpt")

    def test_openai_gpt_41_uses_32768_token_limit(self):
        completions = _RecordingChatCompletionsClient()
        ai_client = object.__new__(UnifiedAIClient)
        ai_client.provider = "openai"
        ai_client.model = "gpt-4.1"
        ai_client.max_tokens = AI_MAX_TOKENS_OPENAI
        ai_client.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        result = ai_client.chat_completion(
            [{"role": "user", "content": "Translate this document."}],
            max_tokens=100_000,
        )

        self.assertEqual(result, "translated")
        self.assertEqual(AI_MAX_TOKENS_OPENAI, 32_768)
        self.assertEqual(completions.kwargs["max_tokens"], 32_768)

    def test_deepseek_chat_stays_at_8192_token_api_limit(self):
        self.assertEqual(AI_MAX_TOKENS_DEEPSEEK, 8_192)
        self.assertEqual(get_provider_max_tokens("deepseek"), 8_192)

    def test_azure_missing_output_text_raises_clear_error(self):
        responses = _RecordingResponsesClient(output_text=None, status="incomplete")
        ai_client = object.__new__(UnifiedAIClient)
        ai_client.provider = "azure"
        ai_client.model = "gpt-5.4"
        ai_client.max_tokens = AI_MAX_TOKENS_AZURE
        ai_client.client = SimpleNamespace(responses=responses)

        with self.assertRaisesRegex(RuntimeError, "did not include output text"):
            ai_client.chat_completion(
                [{"role": "user", "content": "Translate this document."}]
            )

    def test_gemini_receives_temperature_and_output_budget(self):
        recording_models = SimpleNamespace(kwargs=None)

        def generate_content(**kwargs):
            recording_models.kwargs = kwargs
            return SimpleNamespace(
                text=" translated ",
                candidates=[SimpleNamespace(finish_reason="STOP")],
            )

        recording_models.generate_content = generate_content
        ai_client = object.__new__(UnifiedAIClient)
        ai_client.provider = "gemini"
        ai_client.model = "gemini-test"
        ai_client.max_tokens = 4096
        ai_client.client = SimpleNamespace(models=recording_models)

        with mock.patch.object(ai_client_module, "_GEMINI_NEW_SDK", True), mock.patch.object(
            ai_client_module, "_gemini_call_count", 0
        ), mock.patch.object(ai_client_module, "_gemini_cooldown_until", 0.0):
            result = ai_client.chat_completion(
                [{"role": "user", "content": "Translate"}],
                temperature=0.25,
                max_tokens=2048,
            )

        self.assertEqual(result, "translated")
        self.assertEqual(
            recording_models.kwargs["config"],
            {"temperature": 0.25, "max_output_tokens": 2048},
        )

    def test_gemini_max_tokens_finish_reason_is_incomplete_metadata(self):
        models = SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(
                text="partial",
                candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
            )
        )
        ai_client = object.__new__(UnifiedAIClient)
        ai_client.provider = "gemini"
        ai_client.model = "gemini-test"
        ai_client.max_tokens = 1024
        ai_client.client = SimpleNamespace(models=models)

        with mock.patch.object(ai_client_module, "_GEMINI_NEW_SDK", True), mock.patch.object(
            ai_client_module, "_gemini_call_count", 0
        ), mock.patch.object(ai_client_module, "_gemini_cooldown_until", 0.0):
            result = ai_client.chat_completion(
                [{"role": "user", "content": "Translate"}]
            )

        self.assertEqual(result.completion_status, "incomplete")
        self.assertEqual(result.completion_reason, "MAX_TOKENS")


class AIClientCredentialIsolationTest(unittest.TestCase):
    def test_deepseek_rejects_missing_or_empty_credential(self):
        for missing_key in (None, ""):
            with self.subTest(missing_key=missing_key), mock.patch.object(
                ai_client_module, "DEEPSEEK_API_KEY", missing_key
            ), mock.patch("openai.OpenAI") as openai_client:
                with self.assertRaisesRegex(
                    ValueError,
                    "DEEPSEEK_API_TOKEN environment variable must be set",
                ):
                    UnifiedAIClient(provider="deepseek")

                openai_client.assert_not_called()

    def test_deepseek_initializes_without_other_provider_credentials(self):
        with mock.patch.object(
            ai_client_module, "DEEPSEEK_API_KEY", "deepseek-test-key"
        ), mock.patch.object(
            ai_client_module, "GEMINI_API_KEY", None
        ), mock.patch.object(
            ai_client_module, "AZURE_OPENAI_KEY", None
        ), mock.patch(
            "openai.OpenAI"
        ) as openai_client:
            client = UnifiedAIClient(provider="deepseek")

        openai_client.assert_called_once_with(
            api_key="deepseek-test-key",
            base_url=ai_client_module.DEEPSEEK_BASE_URL,
        )
        self.assertEqual(client.provider, "deepseek")

    def test_gemini_initializes_without_other_provider_credentials(self):
        gemini_client = mock.Mock()
        fake_genai = SimpleNamespace(Client=gemini_client)

        with mock.patch.object(
            ai_client_module, "GEMINI_AVAILABLE", True
        ), mock.patch.object(
            ai_client_module, "_GEMINI_NEW_SDK", True
        ), mock.patch.object(
            ai_client_module, "genai", fake_genai
        ), mock.patch.object(
            ai_client_module, "GEMINI_API_KEY", "gemini-test-key"
        ), mock.patch.object(
            ai_client_module, "DEEPSEEK_API_KEY", None
        ), mock.patch.object(
            ai_client_module, "AZURE_OPENAI_KEY", None
        ):
            client = UnifiedAIClient(provider="gemini")

        gemini_client.assert_called_once_with(api_key="gemini-test-key")
        self.assertEqual(client.provider, "gemini")

    def test_azure_initializes_without_other_provider_credentials(self):
        with mock.patch.object(
            ai_client_module, "AZURE_OPENAI_KEY", "azure-test-key"
        ), mock.patch.object(
            ai_client_module,
            "AZURE_OPENAI_BASE_URL",
            "https://azure.example.test/openai/v1",
        ), mock.patch.object(
            ai_client_module, "DEEPSEEK_API_KEY", None
        ), mock.patch.object(
            ai_client_module, "GEMINI_API_KEY", None
        ), mock.patch(
            "openai.OpenAI"
        ) as openai_client:
            client = UnifiedAIClient(provider="azure")

        openai_client.assert_called_once_with(
            api_key="azure-test-key",
            base_url="https://azure.example.test/openai/v1",
        )
        self.assertEqual(client.provider, "azure")


if __name__ == "__main__":
    unittest.main()
