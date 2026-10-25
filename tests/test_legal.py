import shutil
import uuid
from pathlib import Path
from unittest import TestCase

from src.rag.legal.pipeline import ingest_legal, split_frontmatter


SAMPLE = """---
title: "测试法"
office: "测试机关"
publish: "2024-01-01 00:00:00"
effective_date: "2024-02-01 00:00:00"
type: "法律"
status: "有效"
url: "https://example.test/law"
---

# 测试法

第一章 总则

　　第一条　为了测试入库流程，制定本法。

　　第二条　本法适用于测试场景。

第二章 实施

　　第三条　实施时应当遵守本法。
"""


def _workspace_temp() -> Path:
    base = Path.cwd() / "output" / "test_tmp"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"legal-{uuid.uuid4().hex}"
    path.mkdir()
    return path


class FrontmatterTests(TestCase):
    def test_split_frontmatter(self):
        meta, body = split_frontmatter(SAMPLE)
        self.assertEqual(meta["title"], "测试法")
        self.assertEqual(meta["status"], "有效")
        self.assertIn("第一条", body)
        self.assertNotIn("title:", body)


class IngestLegalTests(TestCase):
    def test_article_chunks_keep_metadata(self):
        root = _workspace_temp()
        self.addCleanup(shutil.rmtree, root, True)
        (root / "法律").mkdir()
        (root / "法律" / "测试法.md").write_text(SAMPLE, encoding="utf-8")

        chunks, metrics, details = ingest_legal(root, chunk_size=120)

        self.assertEqual(metrics["selected"], 1)
        self.assertGreaterEqual(metrics["articles"], 3)
        self.assertGreater(len(chunks), 0)
        first = chunks[0]
        self.assertEqual(first.meta["domain"], "legal")
        self.assertEqual(first.meta["title"], "测试法")
        self.assertEqual(first.meta["status"], "有效")
        self.assertTrue(first.meta["article_start"].startswith("第"))
        self.assertIn("测试法", first.text)
        self.assertTrue(any("第一条" in chunk.text for chunk in chunks))
        self.assertEqual(details[0]["title"], "测试法")

    def test_filters_type_and_status(self):
        root = _workspace_temp()
        self.addCleanup(shutil.rmtree, root, True)
        valid = SAMPLE
        invalid_type = SAMPLE.replace('type: "法律"', 'type: "地方性法规"')
        invalid_status = SAMPLE.replace('status: "有效"', 'status: "已废止"')
        (root / "valid.md").write_text(valid, encoding="utf-8")
        (root / "invalid_type.md").write_text(invalid_type, encoding="utf-8")
        (root / "invalid_status.md").write_text(invalid_status, encoding="utf-8")

        chunks, metrics, _ = ingest_legal(root, types=("法律",), statuses=("有效",), chunk_size=120)

        self.assertEqual(metrics["total"], 3)
        self.assertEqual(metrics["skipped_type"], 1)
        self.assertEqual(metrics["skipped_status"], 1)
        self.assertEqual(metrics["selected"], 1)
        self.assertTrue(chunks)

    def test_deduplicates_same_title_and_type(self):
        root = _workspace_temp()
        self.addCleanup(shutil.rmtree, root, True)
        old = SAMPLE.replace("2024-02-01", "2020-02-01")
        (root / "old.md").write_text(old, encoding="utf-8")
        (root / "new.md").write_text(SAMPLE, encoding="utf-8")

        _, metrics, _ = ingest_legal(root, chunk_size=120)

        self.assertEqual(metrics["selected"], 1)
        self.assertEqual(metrics["duplicates"], 1)

    def test_no_article_fallback(self):
        doc = SAMPLE.replace(
            "第一章 总则\n\n　　第一条　为了测试入库流程，制定本法。\n\n"
            "　　第二条　本法适用于测试场景。\n\n第二章 实施\n\n"
            "　　第三条　实施时应当遵守本法。",
            "这是一段没有条文编号的测试正文。\n\n这是第二段测试正文。",
        )
        root = _workspace_temp()
        self.addCleanup(shutil.rmtree, root, True)
        (root / "fallback.md").write_text(doc, encoding="utf-8")

        chunks, metrics, _ = ingest_legal(root, chunk_size=120)

        self.assertEqual(metrics["documents_without_articles"], 1)
        self.assertTrue(chunks)
        self.assertTrue(any("没有条文编号" in chunk.text for chunk in chunks))
