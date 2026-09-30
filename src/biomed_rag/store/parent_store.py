"""JSONL docstore of parent sections used as LLM context after search."""

from __future__ import annotations

from pathlib import Path

from biomed_rag.models.retrieval import ParentChunk


class ParentDocStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, ParentChunk] = {}
        self.load()

    def load(self) -> None:
        self._data = {}
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                chunk = ParentChunk.model_validate_json(line)
                self._data[chunk.parent_id] = chunk

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for chunk in self._data.values():
                handle.write(chunk.model_dump_json() + "\n")

    def upsert(self, chunks: list[ParentChunk]) -> None:
        for chunk in chunks:
            self._data[chunk.parent_id] = chunk
        self.save()

    def get(self, parent_id: str) -> ParentChunk | None:
        return self._data.get(parent_id)

    def mget(self, parent_ids: list[str]) -> list[ParentChunk]:
        out: list[ParentChunk] = []
        seen: set[str] = set()
        for pid in parent_ids:
            if pid in seen:
                continue
            seen.add(pid)
            chunk = self._data.get(pid)
            if chunk:
                out.append(chunk)
        return out

    def __len__(self) -> int:
        return len(self._data)
