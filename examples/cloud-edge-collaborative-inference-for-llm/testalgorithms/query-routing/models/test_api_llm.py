import importlib.util
import sys
import types
import unittest
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


models_pkg = types.ModuleType("models")
models_pkg.__path__ = [str(MODELS_DIR)]
sys.modules.setdefault("models", models_pkg)

load_module("models.base_llm", MODELS_DIR / "base_llm.py")
api_llm = load_module("models.api_llm", MODELS_DIR / "api_llm.py")
APIBasedLLM = api_llm.APIBasedLLM


def build_api_error(status_code, error_type, message):
    class FakeAPIError(Exception):
        __module__ = "openai._exceptions"

        def __init__(self):
            super().__init__(message)
            self.status_code = status_code
            self.body = {
                "error": {
                    "type": error_type,
                    "message": message,
                }
            }

    return FakeAPIError()


class FakeCompletions:
    def __init__(self, error):
        self.error = error

    def create(self, **kwargs):
        raise self.error


class FakeChat:
    def __init__(self, error):
        self.completions = FakeCompletions(error)


class FakeClient:
    def __init__(self, error):
        self.chat = FakeChat(error)


class APIBasedLLMTests(unittest.TestCase):
    @staticmethod
    def build_model(error):
        model = object.__new__(APIBasedLLM)
        model.provider = "openai"
        model.client = FakeClient(error)
        model.model = "gpt-4o-mini"
        model.model_name = "gpt-4o-mini"
        model.temperature = 0.8
        model.max_tokens = 64
        model.top_p = 0.8
        model.repetition_penalty = 1.05
        model.use_cache = False
        model.model_loaded = True
        model.config = {}
        model.is_cache_loaded = True
        return model

    def test_content_policy_failures_return_empty_prediction(self):
        model = self.build_model(
            build_api_error(
                400,
                "content_policy_violation_error",
                "Content validation failed.",
            )
        )

        response = model.inference({"query": "unsafe prompt"})

        self.assertEqual("", response["completion"])
        self.assertIsNone(response["prediction"])
        self.assertEqual(400, response["error"]["status_code"])
        self.assertEqual(
            "content_policy_violation_error",
            response["error"]["type"],
        )

    def test_invalid_requests_still_fail_fast(self):
        model = self.build_model(
            build_api_error(
                400,
                "invalid_request_error",
                "The configured model does not exist.",
            )
        )

        with self.assertRaises(RuntimeError):
            model.inference({"query": "hello"})


if __name__ == "__main__":
    unittest.main()
