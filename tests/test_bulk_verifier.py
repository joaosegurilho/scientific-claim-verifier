from pathlib import Path

from scverifier.core.bulk_verifier import BulkVerifier


class DummyPaper:
    def __init__(self, title: str):
        self.title = title


class DummyKB:
    def get_paper(self, pid):
        # return a simple object with a title attribute
        return DummyPaper(f"Title-{pid}")


def test_load_csv_and_jsonl(tmp_path: Path):
    # CSV with header (dataset_verifier._load_csv currently skips header)
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("id,claim\n1,The sky is blue\n2,Water is wet\n", encoding="utf-8")

    bv = BulkVerifier(dataset_path=csv_path, output_path=tmp_path / "out.csv")
    bv.load()
    assert len(bv.data) == 2
    assert bv.data[0]["id"] == "1"
    assert bv.data[0]["claim"] == "The sky is blue"

    # JSONL
    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text(
        '{"id": "a", "claim": "Alpha"}\n{"id": "b", "claim": "Beta"}\n',
        encoding="utf-8",
    )
    bv_json = BulkVerifier(dataset_path=jsonl_path, output_path=tmp_path / "out.jsonl")
    bv_json.load()
    assert len(bv_json.data) == 2
    assert bv_json.data[1]["id"] == "b"
    assert bv_json.data[1]["claim"] == "Beta"


def test_save_jsonl_txt(tmp_path: Path):
    bv = BulkVerifier(dataset_path=tmp_path / "unused.csv", output_path=tmp_path / "out.csv")
    # Replace KB with dummy implementation to avoid heavy dependencies
    bv.kb = DummyKB()

    # One record; use integer id so TXT formatting that expects numeric IDs works
    bv.data = [
        {
            "id": 1,
            "claim": "C1",
            "verdict": "SUPPORTS",
            "confidence": 0.9,
            "reasoning": "Because...",
            "evidence_sources": ["Title-p1"],
            "evidence_count": 1,
            "token_usage": {"input_tokens": 10, "output_tokens": 20},
            "error": "",
        }
    ]

    # Save CSV
    # bv.processed_index = 0
    # bv._save_csv()
    # out_csv = (tmp_path / "out.csv").read_text(encoding="utf-8")
    # assert "C1" in out_csv
    # assert "Title-p1" in out_csv

    # Save JSONL
    bv.output_path = tmp_path / "out.jsonl"
    bv._save_jsonl()
    out_jsonl = (tmp_path / "out.jsonl").read_text(encoding="utf-8")
    assert '"claim": "C1"' in out_jsonl or "C1" in out_jsonl
    assert "Title-p1" in out_jsonl

    # Save TXT
    bv.output_path = tmp_path / "out.txt"
    bv._save_txt()
    out_txt = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "Claim:" in out_txt
    assert "C1" in out_txt
    assert "Title-p1" in out_txt
