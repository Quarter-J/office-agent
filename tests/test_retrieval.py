import json
from unittest.mock import patch
from tests.support import OfflineTestCase
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_community.vectorstores import FAISS
from pydantic import ValidationError
from src.config.settings import settings
from src.rag import vector_store
from src.tools import search
from src.agent.sub_agents.search_agent import search_agent


class FixedEmbeddings(Embeddings):
    vectors = {"nearest": [0.0, 0.0], "middle": [1.0, 0.0], "distant": [2.0, 0.0],
               "question": [0.0, 0.0], "unrelated": [10.0, 10.0]}

    def embed_documents(self, texts):
        return [self.vectors[text] for text in texts]

    def embed_query(self, text):
        return self.vectors[text]


class RetrievalTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.store = FAISS.from_texts(["distant", "nearest", "middle"], FixedEmbeddings(),
                                     metadatas=[{"source": "distant.md"}, {"source": "nearest.md"}, {"source": "middle.md"}])

    def test_actual_faiss_top_k_orders_nearest_chunks(self):
        with patch.object(vector_store, "load_index", return_value=self.store):
            results = vector_store.search("question", top_k=2, threshold=0)
        self.assertEqual([hit.text for hit in results], ["nearest", "middle"])
        self.assertEqual([hit.score for hit in results], [1.0, 0.5])

    def test_threshold_can_return_fewer_than_k_or_no_results(self):
        with patch.object(vector_store, "load_index", return_value=self.store):
            results = vector_store.search("question", top_k=3, threshold=0.75)
            empty = vector_store.search("unrelated", top_k=3, threshold=0.75)
        self.assertEqual([hit.text for hit in results], ["nearest"])
        self.assertEqual(empty, [])

    def test_invalid_k_rejected_before_retrieval(self):
        with patch.object(vector_store, "load_index") as load:
            for value in (0, -1, 21, 2.5, True, "3"):
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        vector_store.search("question", top_k=value)
                    for tool in (search.knowledge_base_search, search.rag_search):
                        with self.assertRaises(ValidationError):
                            tool.invoke({"query": "question", "top_k": value})
            load.assert_not_called()

    def keyword_fixture(self):
        directory = self.temporary_directory()
        (directory / "guide.md").write_text("Spring Boot 自动装配说明：通过配置和条件注解完成组件自动装配。", encoding="utf-8")
        self.enterContext(patch.object(settings, "knowledge_base_dir", str(directory)))

    def test_keyword_fallback_supplies_rag_context(self):
        self.keyword_fixture()
        model = self.model(AIMessage(content="依据 guide.md，自动装配通过配置和条件注解完成。"))
        with patch.object(vector_store, "search", return_value=[]) as vectors:
            result = json.loads(search.rag_search.invoke({"query": "Spring Boot 自动装配", "top_k": 2}))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["method"], "keyword")
        self.assertEqual(result["sources"], ["guide.md"])
        self.assertIn("条件注解", "\n".join(message.content for message in model.prompts[0]))
        self.assertEqual(vectors.call_count, 1)

    def test_vector_failure_can_still_return_keyword_results(self):
        self.keyword_fixture()
        with patch.object(vector_store, "load_index", side_effect=RuntimeError("index unavailable")):
            result = json.loads(search.knowledge_base_search.invoke({"query": "Spring Boot", "top_k": 1}))
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["method"], "keyword")
        self.assertEqual(len(result["results"]), 1)
        self.assertIn("index unavailable", result["fallback_reason"])

    def test_empty_results_do_not_call_answer_model(self):
        directory = self.temporary_directory()
        model = self.model()
        with patch.object(settings, "knowledge_base_dir", str(directory)), \
             patch.object(vector_store, "search", return_value=[]):
            result = json.loads(search.rag_search.invoke({"query": "没有对应资料"}))
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["sources"], [])
        self.assertEqual(model.prompts, [])

    def test_search_failure_is_not_reported_as_empty(self):
        directory = self.temporary_directory()
        with patch.object(settings, "knowledge_base_dir", str(directory)), \
             patch.object(vector_store, "load_index", side_effect=RuntimeError("broken index")):
            result = json.loads(search.rag_search.invoke({"query": "资料"}))
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "error")
        self.assertIn("broken index", result["error"])

    def test_search_agent_retains_evidence_if_answer_generation_fails(self):
        self.model(RuntimeError("model unavailable"))
        with patch.object(vector_store, "load_index", return_value=self.store) as load:
            result = search_agent({"messages": [HumanMessage(content="question"),
                                                ToolMessage(content="irrelevant tool output", tool_call_id="d")],
                                   "intent": "knowledge_search"})
        self.assertEqual(result["task_status"], "failed")
        self.assertIn("nearest.md", result["final_output"])
        self.assertIn("nearest", result["final_output"])
        self.assertEqual(load.call_count, 1)

    def test_vector_rag_and_snippet_tools_share_retrieval_fields(self):
        self.model(AIMessage(content="根据 nearest.md 回答"))
        with patch.object(vector_store, "load_index", return_value=self.store):
            snippets = json.loads(search.knowledge_base_search.invoke({"query": "question", "top_k": 1}))
            rag = json.loads(search.rag_search.invoke({"query": "question", "top_k": 1}))
        self.assertEqual(snippets["results"], rag["results"])
        self.assertEqual(set(rag["results"][0]), {"rank", "source", "score", "content"})
