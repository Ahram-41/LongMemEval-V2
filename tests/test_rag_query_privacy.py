import unittest

from evaluation.harness import build_prompt_row
from memory_modules.agentrunbook_r import AgentRunbookR
from memory_modules.memory import Memory


class QueryContextSpyMemory(Memory):
    def __init__(self, memory_type: str) -> None:
        self.memory_type = memory_type
        self.seen_query_context: dict[str, object] | None = None
        super().__init__({})

    def insert(self, trajectory: dict[str, object]) -> None:
        _ = trajectory

    def query(self, query: str, query_image: str | None = None) -> list[dict[str, str]]:
        _ = query, query_image
        self.seen_query_context = self.get_query_context()
        return []


def prompt_item() -> dict[str, object]:
    return {
        "index": 0,
        "question_item": {
            "id": "secret-question-id",
            "question_type": "procedure-abs",
            "question": "How do I complete the task?",
            "answer": "secret-answer",
            "eval_function": "secret-evaluator",
            "metadata": {"original_goal": ["secret-original-goal"]},
        },
        "question_id": "secret-question-id",
        "question_type": "procedure-abs",
        "category": "procedure-abs",
        "eval_function": "secret-evaluator",
        "eval_name": "llm_abstention_checker",
        "question_text": "How do I complete the task?",
        "question_image": None,
        "answer_gold": "secret-answer",
    }


class RagQueryPrivacyTest(unittest.TestCase):
    def test_rag_backends_receive_empty_query_context(self) -> None:
        for memory_type in ("rag", "agentrunbook_r"):
            with self.subTest(memory_type=memory_type):
                memory = QueryContextSpyMemory(memory_type)
                build_prompt_row(
                    prompt_item(),
                    haystack_ids=[],
                    memory=memory,
                    system_prompt="Answer using memory.",
                    memory_context_max_tokens=1,
                )
                self.assertEqual(memory.seen_query_context, {})

    def test_agentrunbook_r_planner_uses_only_question_text(self) -> None:
        memory = object.__new__(AgentRunbookR)
        memory._runtime_query_summary = "safe memory summary"

        messages = memory._build_query_generation_messages(
            query="How do I complete the task?",
        )
        prompt = messages[1]["content"]

        self.assertIn("Question text: How do I complete the task?", prompt)
        for forbidden in (
            "Question ID:",
            "Question type:",
            "Question image path:",
            "Original goals attached",
        ):
            self.assertNotIn(forbidden, prompt)

    def test_agentrunbook_r_fallback_is_metadata_independent(self) -> None:
        memory = object.__new__(AgentRunbookR)
        self.assertEqual(
            memory._fallback_query_bundle(query="  Find   the relevant workflow  "),
            {
                "raw_state_queries": ["Find the relevant workflow"],
                "event_query": "Find the relevant workflow",
                "note_query": "Find the relevant workflow",
            },
        )


if __name__ == "__main__":
    unittest.main()
