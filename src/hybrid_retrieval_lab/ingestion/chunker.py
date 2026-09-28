"""Transform extracted source pages into stable, metadata-rich retrieval chunks."""

from __future__ import annotations

import textwrap
from collections.abc import Iterator, Sequence
from typing import TypedDict


class Section(TypedDict):
    heading: str
    blocks: list[str]


class Page(TypedDict):
    id: str
    title: str
    url: str
    sections: list[Section]


class Chunk(TypedDict):
    id: str
    source_id: str
    source_url: str
    title: str
    section: str
    text: str


MAX_CHARS = 1600


def _wrap_block(block: str, max_chars: int) -> list[str]:
    return textwrap.wrap(
        block,
        width=max_chars,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _iter_wrapped_blocks(blocks: Sequence[str], max_chars: int) -> Iterator[str]:
    for block in blocks:
        yield from _wrap_block(block, max_chars)


def _make_chunk(page: Page, section: Section, blocks: list[str], ordinal: int) -> Chunk:
    return {
        "id": f"{page['id']}-{ordinal:03d}",
        "source_id": page["id"],
        "source_url": page["url"],
        "title": page["title"],
        "section": section["heading"],
        "text": "\n\n".join(blocks),
    }


def _chunks_for_section(page: Page, section: Section, first_ordinal: int, max_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    group: list[str] = []
    size = 0
    for block in _iter_wrapped_blocks(section["blocks"], max_chars):
        if group and size + len(block) + 2 > max_chars:
            chunks.append(_make_chunk(page, section, group, first_ordinal + len(chunks)))
            group = []
            size = 0
        group.append(block)
        size += len(block) + 2
    if group:
        chunks.append(_make_chunk(page, section, group, first_ordinal + len(chunks)))
    return chunks


def _chunks_for_page(page: Page, max_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in page["sections"]:
        first_ordinal = len(chunks) + 1
        chunks.extend(_chunks_for_section(page, section, first_ordinal, max_chars))
    return chunks


def chunk_pages(pages: Sequence[Page], max_chars: int = MAX_CHARS) -> list[Chunk]:
    """Chunk pages in source order while retaining source and section metadata."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(_chunks_for_page(page, max_chars))
    return chunks
