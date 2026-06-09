import csv
import json
import sys
from itertools import islice
from pathlib import Path
from typing import List

from scverifier.config.settings import Config
from scverifier.core.agents.autonomous_agent import AutonomousClaimAgent
from scverifier.core.benchmarking.base import VerificationMethod
from scverifier.core.knowledge.knowledge_base import KnowledgeBase
from scverifier.pipelines.verification_pipeline import VerificationPipeline


# TODO: Finda better way to progressively write
# TODO: Add csv suport (think about it better) in the test too
# TODO: Add support for  skip_extraction_eval and use_all_propositions
class DatasetVerifier:
    def __init__(
        self,
        dataset_path: Path,
        output_path: Path | None = None,
        claim_column: int = 1,
        id_column: int = 0,
        method: VerificationMethod = VerificationMethod.AGENTLESS_WITH_SEARCH,
    ):
        self.dataset_path: Path = dataset_path
        self.output_path: Path | None = (
            output_path if output_path else dataset_path.parent / f"{dataset_path.stem}_verified.jsonl"
        )
        self.claim_column: int = claim_column
        self.id_column: int | None = id_column  # Set to None if using index as ID
        self.method: VerificationMethod = method

        self.kb: KnowledgeBase = KnowledgeBase()
        self.data: List = []
        # Number of items that have been processed (i.e. verified). Use int count.
        self.processed_index: int = 0
        # Number of records already written to the output file. Used to append only new records.
        # self._saved_count: int = 0

    def load(self, max_items: int | None = None):
        """
        Load the dataset and prepare it for verification.

        Args:
            max_items (int | None): Maximum number of items to load from the dataset. If None, load all items.
        """
        ext = self.dataset_path.suffix.lower()
        if ext == ".csv":
            self.data = self._load_csv(max_items)
        elif ext == ".tsv":
            self.data = self._load_csv(max_items, sep="\t")
        elif ext == ".jsonl":
            self.data = self._load_jsonl(max_items)
        else:
            raise ValueError(f"Unsupported dataset format: {ext}")

    def _load_csv(self, max_items: int | None = None, sep: str = ","):
        """
        Load a CSV or TSV dataset.
        """
        records = []
        with open(self.dataset_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=sep)

            def extract_row(row):
                return {
                    "id": str(row[self.id_column]),
                    "claim": row[self.claim_column],
                    "row": row,
                }

            for row in islice(reader, 1, max_items):
                records.append(extract_row(row))

        return records

    def _load_jsonl(self, max_items: int | None = None):
        """
        Load a JSONL dataset.
        """
        records = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(islice(f, max_items)):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rec_id = str(obj.get("id", idx))
                    claim = obj.get("claim", "")
                    records.append({"id": rec_id, "claim": claim, "row": obj})
                else:
                    records.append({"id": str(idx), "claim": "", "row": obj})

        return records

    def run(
        self,
        kb_only: bool = False,
        skip_extraction_eval: bool = False,
        use_all_propositions: bool = False,
        max_papers: int = 10,
        max_items: int | None = None,
        resume: bool = False,
    ):
        """
        Verify all claims in the dataset.
        """

        use_search = self.method in (
            VerificationMethod.AGENT_WITH_SEARCH,
            VerificationMethod.AGENTLESS_WITH_SEARCH,
        )
        use_agent = self.method in (
            VerificationMethod.AGENT,
            VerificationMethod.AGENT_WITH_SEARCH,
        )

        # Initialize KB
        print("\n Loading existing knowledge base...")
        try:
            self.kb.load()
            print(f"   Knowledge base loaded from {Config.DB_NAME}")
        except FileNotFoundError:
            if kb_only:
                print(f"  Error: No knowledge base found at {Config.DB_NAME}")
                print("   Cannot use --kb-only mode without an existing knowledge base.")
                print("   Run without --kb-only to search for papers first.")
                sys.exit(1)
            else:
                print(f"  No existing knowledge base found at {Config.DB_NAME}")
                print("   Starting with fresh knowledge base.")
        except Exception as e:
            print(f"     Error loading knowledge base: {e}")
            if kb_only:
                print("   Cannot proceed in --kb-only mode.")
                sys.exit(1)
            print("   Starting with fresh knowledge base.")

        # Initialize verifier based on method
        pipeline = VerificationPipeline(kb=self.kb) if not use_agent else None
        agent = AutonomousClaimAgent(kb=self.kb, allow_online_search=use_search) if use_agent else None

        self.load(max_items=max_items)
        if resume:
            verified_claims = set(self._get_verified_claims())
        else:
            verified_claims = set()

        for i, item in enumerate(self.data):
            claim_id = item.get("id", str(i))
            if claim_id in verified_claims:
                continue
            claim_text = item.get("claim", "")

            print(f"Verifying claim {claim_id}: {claim_text}")

            try:
                if use_agent and agent is not None:
                    result = agent.verify_claim(
                        claim_text,
                    )
                elif pipeline is not None:
                    result = pipeline.verify_claim_with_search(
                        claim_text,
                        max_papers=max_papers,
                        # skip_extraction_eval=skip_extraction_eval,
                        # use_all_propositions=use_all_propositions,
                    )
                else:
                    raise ValueError("No verification method selected.")

                self.data[i].update(result.to_dict())

            except KeyboardInterrupt:
                print(" Interrupted verification. Finishing save...", end="\t")
                self.save()
                print("DONE.")

            except Exception as e:
                print(f"Error verifying claim {claim_id}: {e}")
                self.data[i]["error"] = str(e)

            # Update processed count then save so only new records are written
            self.processed_index = i + 1
            try:
                self.save()
            except Exception as e:
                print(f"Error saving progress after claim {claim_id}: {e}")
            # Track how many were saved
            # self._saved_count = self.processed_index

    def _get_verified_claims(self):
        verified_claims = []
        ext = self.output_path.suffix.lower()
        with open(self.output_path, "r") as f:
            if ext == ".txt":
                for line in f.readlines():
                    if line.startswith("====="):
                        claim = int(  # Use int to remove leading zeros
                            line.removeprefix("===== Claim <").removesuffix("> =====\n")
                        )
                        verified_claims.append(str(claim))

            # elif ext == ".csv":
            #     csv_reader = csv.reader(f)
            #     for i, row in enumerate(csv_reader):
            #         if i == 0:
            #             continue
            #         if row[2] is not None:
            #             claim = str(row[0])
            #             verified_claims.append(claim)

            # elif ext == ".tsv":
            #     tsv_reader = csv.reader(f, delimiter="\t")
            #     for i, row in enumerate(tsv_reader):
            #         if i == 0:
            #             continue
            #         if row[2] is not None:
            #             claim = str(row[0])
            #             verified_claims.append(claim)

            elif ext == ".jsonl":
                for line in f.readlines():
                    claim = str(json.loads(line).get("id"))
                    verified_claims.append(claim)

        return verified_claims

    def save(self):
        """
        Save the verification results to the output path.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        ext = self.output_path.suffix.lower()
        # if ext == ".csv":
        #     self._save_csv()
        # elif ext == ".tsv":
        #     self._save_csv(sep="\t")
        if ext == ".jsonl":
            self._save_jsonl()
        elif ext == ".txt":
            self._save_txt()
        else:
            raise ValueError(f"Unsupported output format: {ext}")

    def _get_evidence_titles(self, record):
        """
        Return a list of evidence titles for a record.
        Prefer `evidence_sources` if present (list or string), otherwise look up
        titles from `papers_used` via the knowledge base.
        """
        titles = []
        es = record.get("evidence_sources")
        if es:
            if isinstance(es, list):
                titles = [str(x) for x in es]
            else:
                titles = [str(es)]
        else:
            for pid in record.get("papers_used", []):
                try:
                    paper = self.kb.get_paper(pid)
                    titles.append(paper.title if paper else str(pid))
                except Exception:
                    titles.append(str(pid))
        return titles

    # def _save_csv(self, sep=","):
    #     """
    #     Save the verification results in CSV format.
    #     """
    #     mode = "w"
    #     with open(self.output_path, mode, newline="", encoding="utf-8") as f:
    #         writer = csv.writer(f, delimiter=sep)
    #         # Write header
    #         if self.processed_index == 0:
    #             writer.writerow(
    #                 [
    #                     "id",
    #                     "claim",
    #                     "verdict",
    #                     "confidence",
    #                     "reasoning",
    #                     "evidence_sources",
    #                     "evidence_count",
    #                     "token_input",
    #                     "token_output",
    #                     "error",
    #                 ]
    #             )
    #         # Write data rows
    #         for record in islice(self.data, self.processed_index + 1):
    #             titles = self._get_evidence_titles(record)
    #             evidence_count = record.get(
    #                 "evidence_count", record.get("num_evidence", "")
    #             )
    #             writer.writerow(
    #                 [
    #                     record["id"],
    #                     record["claim"],
    #                     record.get("verdict", ""),
    #                     record.get("confidence", ""),
    #                     record.get("reasoning", ""),
    #                     "; ".join(titles),
    #                     evidence_count,
    #                     record.get("token_usage", {}).get("input_tokens", ""),
    #                     record.get("token_usage", {}).get("output_tokens", ""),
    #                     record.get("error", ""),
    #                 ]
    #             )

    def _save_jsonl(self):
        """
        Save the verification results in JSONL format.
        """
        mode = "w"
        with open(self.output_path, mode, encoding="utf-8") as f:
            for record in islice(self.data, self.processed_index + 1):
                titles = self._get_evidence_titles(record)
                rec = {
                    "id": record["id"],
                    "claim": record["claim"],
                    "verdict": record.get("verdict", ""),
                    "confidence": record.get("confidence", ""),
                    "reasoning": record.get("reasoning", ""),
                    "evidence_sources": "; ".join(titles),
                    "evidence_count": record.get("evidence_count", record.get("num_evidence", "")),
                    "token_input": record.get("token_usage", {}).get("input_tokens", ""),
                    "token_output": record.get("token_usage", {}).get("output_tokens", ""),
                    "error": record.get("error", ""),
                }

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _save_txt(self):
        """
        Save the verification results in plain text format.
        """
        mode = "w"
        with open(self.output_path, mode, encoding="utf-8") as f:
            for record in islice(self.data, self.processed_index + 1):
                # Preserve zero-padded numeric IDs when possible
                rid = record.get("id", "")
                try:
                    rid_str = f"{int(rid):03d}"
                except Exception:
                    rid_str = str(rid)
                f.write(f"===== Claim <{rid_str}> =====\n")
                f.write(f"Claim: {record['claim']}\n")
                f.write(f"Verdict: {record.get('verdict', '')}\n")
                f.write(f"Confidence: {record.get('confidence', '')}\n")
                f.write(f"Reasoning: {record.get('reasoning', '')}\n")
                titles = self._get_evidence_titles(record)
                f.write(f"Evidence Sources: {', '.join(titles)}\n")
                f.write(f"Error: {record.get('error', '')}\n")
                f.write("-" * 80 + "\n")
