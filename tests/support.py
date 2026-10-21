import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

# Test processes must not load real model/Redis credentials or append app logs.
os.environ.update(OPENAI_API_KEY="offline-test", REDIS_URL="", MEMORY_ENABLED="false")
logging.getLogger("office_agent").addHandler(logging.NullHandler())


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.prompts.append(messages)
        if not self.responses:
            raise AssertionError("Unexpected extra model call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("Network is disabled in tests")))
        from src.config.settings import settings
        ledger = str(self.temporary_directory() / "actions.db")
        self.enterContext(patch.object(settings, "action_ledger_path", ledger))
        self.enterContext(patch.dict(os.environ, {"ACTION_LEDGER_PATH": ledger}))

    def temporary_directory(self):
        root = Path(__file__).resolve().parents[1]
        temp = tempfile.TemporaryDirectory(prefix=".office-agent-test-", dir=str(root))
        target = Path(temp.name).resolve()
        assert target.parent == root and target.name.startswith(".office-agent-test-")
        self.addCleanup(temp.cleanup)
        return target

    def model(self, *responses):
        from src.api import llm
        model = ScriptedLLM(*responses)
        self.enterContext(patch.object(llm, "get_llm", return_value=model))
        return model
