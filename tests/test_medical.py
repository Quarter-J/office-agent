import shutil
import uuid
from pathlib import Path
from unittest import TestCase

from src.rag.medical.normalize import full_to_half, normalize_text
from src.rag.medical.terminology import TerminologyMapper
from src.rag.medical.dirty import detect_dirty
from src.rag.medical.pipeline import ingest_medical


class NormalizeTests(TestCase):
    def test_full_to_half(self):
        self.assertEqual(full_to_half("ＡＢＣ１２３（）："), "ABC123():")

    def test_normalize_units(self):
        out = normalize_text("每次 100 毫克")
        self.assertIn("mg", out)
        self.assertNotIn("毫克", out)

    def test_collapse_whitespace(self):
        self.assertEqual(normalize_text("  第一行   \n   第二行  "), "第一行\n第二行")


class TerminologyTests(TestCase):
    def setUp(self):
        self.mapper = TerminologyMapper({"阿司匹林": ["乙酰水杨酸", "Aspirin"], "恶心": ["反胃"]})

    def test_map_term(self):
        self.assertEqual(self.mapper.map_term("乙酰水杨酸"), "阿司匹林")
        self.assertEqual(self.mapper.map_term("aspirin"), "阿司匹林")
        self.assertEqual(self.mapper.map_term("布洛芬"), "布洛芬")

    def test_map_text_longest_first(self):
        self.assertEqual(self.mapper.map_text("服用乙酰水杨酸后出现反胃"), "服用阿司匹林后出现恶心")


class DirtyTests(TestCase):
    def test_empty(self):
        ok, reason = detect_dirty("   \n  ")
        self.assertTrue(ok)
        self.assertEqual(reason, "empty")

    def test_too_short(self):
        ok, reason = detect_dirty("阿司匹林")
        self.assertTrue(ok)
        self.assertEqual(reason, "too_short")

    def test_garbled(self):
        ok, reason = detect_dirty("药品说明书内容因编码损坏出现�乱码，无法正常解析入库。")
        self.assertTrue(ok)
        self.assertEqual(reason, "garbled")

    def test_clean(self):
        ok, reason = detect_dirty("阿司匹林肠溶片用于解热镇痛，缓解轻至中度疼痛。")
        self.assertFalse(ok)
        self.assertIsNone(reason)


class IngestMedicalTests(TestCase):
    def test_ingest_metrics(self):
        base = Path.cwd() / "output" / "test_tmp"
        base.mkdir(parents=True, exist_ok=True)
        root = base / f"medical-{uuid.uuid4().hex}"
        root.mkdir()
        self.addCleanup(shutil.rmtree, root, True)
        (root / "clean1.md").write_text(
            "# 药品\n\n## 适应症\n\n用于解热镇痛，缓解疼痛症状。", encoding="utf-8")
        (root / "empty.txt").write_text("", encoding="utf-8")
        (root / "short.txt").write_text("阿司匹林", encoding="utf-8")

        mapper = TerminologyMapper({"阿司匹林": ["乙酰水杨酸"]})
        chunks, metrics, details = ingest_medical(root, mapper)

        self.assertEqual(metrics["total"], 3)
        self.assertEqual(metrics["intercepted"], 2)
        self.assertEqual(metrics["ingested"], 1)
        self.assertEqual(metrics["intercept_rate"], round(2 / 3, 4))
        self.assertGreater(metrics["chunks"], 0)
        self.assertEqual([d["status"] for d in details], ["ingested", "intercepted", "intercepted"])
