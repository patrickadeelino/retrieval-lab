"""Transform extracted source pages into stable, metadata-rich retrieval chunks."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
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


def _iter_token_bounded_blocks(
    blocks: Sequence[str], max_tokens: int, token_count: Callable[[str], int]
) -> Iterator[str]:
    for block in blocks:
        if token_count(block) <= max_tokens:
            yield block
            continue
        yield from _split_block_by_tokens(block, max_tokens, token_count)


def _split_block_by_tokens(block: str, max_tokens: int, token_count: Callable[[str], int]) -> Iterator[str]:
    words = block.split()
    group: list[str] = []
    for word in words:
        if token_count(word) > max_tokens:
            if group:
                yield " ".join(group)
                group = []
            # Keep an indivisible over-budget word isolated. The indexer records and
            # skips that one chunk rather than losing the rest of the source page.
            yield word
            continue
        candidate = " ".join((*group, word))
        if group and token_count(candidate) > max_tokens:
            yield " ".join(group)
            group = [word]
        else:
            group.append(word)
    if group:
        yield " ".join(group)


def _make_chunk(page: Page, section: Section, blocks: list[str], ordinal: int) -> Chunk:
    return {
        "id": f"{page['id']}-{ordinal:03d}",
        "source_id": page["id"],
        "source_url": page["url"],
        "title": page["title"],
        "section": section["heading"],
        "text": "\n\n".join(blocks),
    }


def _chunks_for_section(
    page: Page,
    section: Section,
    first_ordinal: int,
    max_tokens: int,
    token_count: Callable[[str], int],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    group: list[str] = []
    for block in _iter_token_bounded_blocks(section["blocks"], max_tokens, token_count):
        candidate = "\n\n".join((*group, block))
        exceeds_tokens = group and token_count(candidate) > max_tokens
        if exceeds_tokens:
            chunks.append(_make_chunk(page, section, group, first_ordinal + len(chunks)))
            group = []
        group.append(block)
    if group:
        chunks.append(_make_chunk(page, section, group, first_ordinal + len(chunks)))
    return chunks


def _chunks_for_page(page: Page, max_tokens: int, token_count: Callable[[str], int]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in page["sections"]:
        first_ordinal = len(chunks) + 1
        chunks.extend(_chunks_for_section(page, section, first_ordinal, max_tokens, token_count))
    return chunks


def chunk_pages(
    pages: Sequence[Page],
    token_count: Callable[[str], int],
    max_tokens: int,
) -> list[Chunk]:
    """Chunk pages to the model-token budget while retaining source metadata."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(_chunks_for_page(page, max_tokens, token_count))
    return chunks
