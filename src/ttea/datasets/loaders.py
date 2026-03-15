from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..exceptions import DatasetUnavailableError
from .registry import DatasetDescriptor


class BaseDatasetLoader:
    def __init__(self, descriptor: DatasetDescriptor) -> None:
        self.descriptor = descriptor

    def is_available(self) -> bool:
        return any((self.descriptor.local_path / filename).exists() for filename in self.descriptor.expected_files)

    def ensure_available(self) -> None:
        if not self.is_available():
            raise DatasetUnavailableError(
                f"Dataset {self.descriptor.name} is not available under {self.descriptor.local_path}. "
                f"Source: {self.descriptor.acquisition_url}"
            )

    def load_records(self, split: str = "test", limit: int | None = None, **_: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _limit(self, records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
        return records if limit is None else records[:limit]


class StructuredJsonLoader(BaseDatasetLoader):
    def load_records(self, split: str = "test", limit: int | None = None, **_: Any) -> list[dict[str, Any]]:
        self.ensure_available()
        candidate_names = [
            f"{split}.jsonl",
            f"{split}.json",
            "records.jsonl",
            "records.json",
            "tasks.jsonl",
            "tasks.json",
        ]
        for filename in candidate_names:
            path = self.descriptor.local_path / filename
            if path.exists():
                return self._limit(self._read(path), limit)
        raise DatasetUnavailableError(f"No readable record file found for {self.descriptor.name} in {self.descriptor.local_path}")

    def _read(self, path: Path) -> list[dict[str, Any]]:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if "records" in payload and isinstance(payload["records"], list):
                return payload["records"]
            if "data" in payload and isinstance(payload["data"], list):
                return payload["data"]
        raise DatasetUnavailableError(f"Unsupported JSON structure in {path}")


class ParallelTextLoader(BaseDatasetLoader):
    def load_records(
        self,
        split: str = "test",
        limit: int | None = None,
        source_language: str = "en",
        target_language: str = "de",
        **_: Any,
    ) -> list[dict[str, Any]]:
        self.ensure_available()
        source_path = self.descriptor.local_path / f"{split}.{source_language}"
        target_path = self.descriptor.local_path / f"{split}.{target_language}"
        if not source_path.exists() or not target_path.exists():
            raise DatasetUnavailableError(
                f"Expected paired files {source_path.name} and {target_path.name} under {self.descriptor.local_path}"
            )
        with source_path.open("r", encoding="utf-8") as source_handle:
            source_lines = [line.rstrip("\n") for line in source_handle]
        with target_path.open("r", encoding="utf-8") as target_handle:
            target_lines = [line.rstrip("\n") for line in target_handle]
        pairs = []
        for index, (source_text, target_text) in enumerate(zip(source_lines, target_lines)):
            pairs.append(
                {
                    "id": f"{split}-{source_language}-{target_language}-{index}",
                    "source_text": source_text,
                    "target_text": target_text,
                    "source_language": source_language,
                    "target_language": target_language,
                }
            )
        return self._limit(pairs, limit)


class SQuADLoader(BaseDatasetLoader):
    def load_records(self, split: str = "dev", limit: int | None = None, **_: Any) -> list[dict[str, Any]]:
        self.ensure_available()
        candidate_names = [f"{split}.json", "train.json", "dev.json", "test.json"]
        for filename in candidate_names:
            path = self.descriptor.local_path / filename
            if path.exists():
                return self._limit(self._flatten(path), limit)
        raise DatasetUnavailableError(f"No SQuAD JSON file found in {self.descriptor.local_path}")

    def _flatten(self, path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records: list[dict[str, Any]] = []
        data = payload.get("data", payload if isinstance(payload, list) else [])
        for article in data:
            for paragraph in article.get("paragraphs", []):
                context = paragraph.get("context", "")
                for qa in paragraph.get("qas", []):
                    answers = qa.get("answers", [])
                    records.append(
                        {
                            "id": qa.get("id"),
                            "question": qa.get("question", ""),
                            "context": context,
                            "answer": answers[0]["text"] if answers else "",
                            "answers": [answer.get("text", "") for answer in answers if answer.get("text")],
                        }
                    )
        return records


class DatasetLoaderFactory:
    @staticmethod
    def create(descriptor: DatasetDescriptor) -> BaseDatasetLoader:
        lowered = descriptor.name.lower()
        if lowered == "jrc-acquis":
            return ParallelTextLoader(descriptor)
        if lowered == "squad":
            return SQuADLoader(descriptor)
        return StructuredJsonLoader(descriptor)
