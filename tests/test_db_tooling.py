import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.queries.db_tooling import run_llm_db_session


class FakeCompletions:
    def __init__(self, responses):
        self._responses = responses
        self._idx = 0

    def create(self, **kwargs):
        response = self._responses[self._idx]
        self._idx += 1
        return response


class FakeChat:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)


class FakeClient:
    def __init__(self, responses):
        self.chat = FakeChat(responses)


def _tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response_with_tool_call(call_id: str, sql: str):
    message = SimpleNamespace(
        content="Ich pruefe die Datenbank.",
        tool_calls=[_tool_call(call_id, "execute_sql", '{"sql": "%s"}' % sql)],
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _response_with_named_tool_call(call_id: str, name: str, arguments: str):
    message = SimpleNamespace(
        content="Tool call requested.",
        tool_calls=[_tool_call(call_id, name, arguments)],
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _response_with_text(text: str):
    message = SimpleNamespace(content=text, tool_calls=[])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class DbToolingLoopTests(unittest.TestCase):
    def test_tool_call_then_final_answer(self):
        responses = [
            _response_with_tool_call("call_1", "SELECT 1 AS n"),
            _response_with_text("Die Abfrage wurde erfolgreich ausgefuehrt."),
        ]
        client = FakeClient(responses)

        calls = []

        def fake_executor(sql: str):
            calls.append(sql)
            return {"ok": True, "rows": [{"n": 1}]}

        final_text = run_llm_db_session(
            client=client,
            user_prompt="Fuehre einen Test aus.",
            sql_executor=fake_executor,
            max_steps=3,
        )

        self.assertEqual(calls, ["SELECT 1 AS n"])
        self.assertIn("erfolgreich", final_text)

    def test_extra_tool_handler_is_used(self):
        responses = [
            _response_with_named_tool_call(
                "call_2",
                "lookup_omop_concepts",
                '{"term": "heart attack", "domain_hint": "Condition", "top_k": 3}',
            ),
            _response_with_text("Mapping completed."),
        ]
        client = FakeClient(responses)

        lookup_calls = []

        def fake_lookup(args: dict):
            lookup_calls.append(args)
            return {
                "ok": True,
                "row_count": 1,
                "best_candidate": {
                    "concept_id": 4329847,
                    "concept_name": "Myocardial infarction",
                    "domain_id": "Condition",
                },
            }

        final_text = run_llm_db_session(
            client=client,
            user_prompt="Normalize terms.",
            extra_tool_handlers={"lookup_omop_concepts": fake_lookup},
            max_steps=3,
        )

        self.assertEqual(len(lookup_calls), 1)
        self.assertEqual(lookup_calls[0]["term"], "heart attack")
        self.assertIn("Mapping", final_text)


if __name__ == "__main__":
    unittest.main()

