from pathlib import Path

from scverifier.core.dataset_verifier import DatasetVerifier


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

    dv = DatasetVerifier(dataset_path=csv_path, output_path=tmp_path / "out.csv")
    dv.load()
    assert len(dv.data) == 2
    assert dv.data[0]["id"] == "1"
    assert dv.data[0]["claim"] == "The sky is blue"

    # JSONL
    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text(
        '{"id": "a", "claim": "Alpha"}\n{"id": "b", "claim": "Beta"}\n',
        encoding="utf-8",
    )
    dv_json = DatasetVerifier(dataset_path=jsonl_path, output_path=tmp_path / "out.jsonl")
    dv_json.load()
    assert len(dv_json.data) == 2
    assert dv_json.data[1]["id"] == "b"
    assert dv_json.data[1]["claim"] == "Beta"


def test_save_jsonl_txt(tmp_path: Path):
    dv = DatasetVerifier(dataset_path=tmp_path / "unused.csv", output_path=tmp_path / "out.csv")
    # Replace KB with dummy implementation to avoid heavy dependencies
    dv.kb = DummyKB()

    # One record; use integer id so TXT formatting that expects numeric IDs works
    dv.data = [
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
    # dv.processed_index = 0
    # dv._save_csv()
    # out_csv = (tmp_path / "out.csv").read_text(encoding="utf-8")
    # assert "C1" in out_csv
    # assert "Title-p1" in out_csv

    # Save JSONL
    dv.output_path = tmp_path / "out.jsonl"
    dv._save_jsonl()
    out_jsonl = (tmp_path / "out.jsonl").read_text(encoding="utf-8")
    assert '"claim": "C1"' in out_jsonl or "C1" in out_jsonl
    assert "Title-p1" in out_jsonl

    # Save TXT
    dv.output_path = tmp_path / "out.txt"
    dv._save_txt()
    out_txt = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "Claim:" in out_txt
    assert "C1" in out_txt
    assert "Title-p1" in out_txt
