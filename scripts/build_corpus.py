"""Build a small, frozen retrieval corpus from selected Portuguese GitHub Docs pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import textwrap
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from bs4 import BeautifulSoup
from bs4.element import Tag


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


SOURCES = {
    "webhook-best-practices": "https://docs.github.com/pt/webhooks/using-webhooks/best-practices-for-using-webhooks",
    "webhook-validation": "https://docs.github.com/pt/webhooks/using-webhooks/validating-webhook-deliveries",
    "webhook-failures": "https://docs.github.com/pt/webhooks/using-webhooks/handling-failed-webhook-deliveries",
    "rest-pagination": "https://docs.github.com/pt/rest/using-the-rest-api/using-pagination-in-the-rest-api",
    "rest-rate-limits": "https://docs.github.com/pt/rest/using-the-rest-api/rate-limits-for-the-rest-api",
    "rest-troubleshooting": "https://docs.github.com/pt/rest/using-the-rest-api/troubleshooting-the-rest-api",
}
MAX_CHARS = 1600


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fetch(url: str) -> bytes:
    return subprocess.run(
        ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "30", url],
        check=True,
        capture_output=True,
    ).stdout


def extract(html: bytes, source_id: str, url: str) -> Page:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one("#article-contents .markdown-body")
    if body is None:
        raise ValueError(f"Article body missing: {url}")
    heading = soup.select_one("h1")
    title = heading.get_text(" ", strip=True) if heading else source_id
    sections: list[Section] = []
    section = Section(heading="Introduction", blocks=[])
    skip_section = False
    for element in body.children:
        if not isinstance(element, Tag):
            continue
        if element.name in {"h2", "h3"}:
            if section["blocks"]:
                sections.append(section)
            heading_text = element.get_text(" ", strip=True)
            section = Section(heading=heading_text, blocks=[])
            skip_section = heading_text.lower().startswith(
                (
                    "further reading",
                    "leitura adicional",
                    "example",
                    "exemplo",
                    "redelivering organization",
                    "redelivering github app",
                    "redelivering github marketplace",
                    "redelivering github sponsors",
                )
            )
            continue
        if skip_section or element.name not in {"p", "ul", "ol", "table", "div"}:
            continue
        for noise in element.select("button, svg, script"):
            noise.decompose()
        elements = element.find_all("li", recursive=False) if element.name in {"ul", "ol"} else [element]
        for item in elements:
            block = " ".join(item.stripped_strings)
            if block:
                section["blocks"].append(block)
    if section["blocks"]:
        sections.append(section)
    return {"id": source_id, "title": title, "url": url, "sections": sections}


def make_chunks(pages: list[Page]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        ordinal = 0
        for section in page["sections"]:
            group: list[str] = []
            size = 0
            for block in (
                part
                for original in section["blocks"]
                for part in textwrap.wrap(original, width=MAX_CHARS, break_long_words=False, break_on_hyphens=False)
            ):
                if group and size + len(block) + 2 > MAX_CHARS:
                    ordinal += 1
                    chunks.append(
                        {
                            "id": f"{page['id']}-{ordinal:03d}",
                            "source_id": page["id"],
                            "source_url": page["url"],
                            "title": page["title"],
                            "section": section["heading"],
                            "text": "\n\n".join(group),
                        }
                    )
                    group, size = [], 0
                group.append(block)
                size += len(block) + 2
            if group:
                ordinal += 1
                chunks.append(
                    {
                        "id": f"{page['id']}-{ordinal:03d}",
                        "source_id": page["id"],
                        "source_url": page["url"],
                        "title": page["title"],
                        "section": section["heading"],
                        "text": "\n\n".join(group),
                    }
                )
    return chunks


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def read_pages(path: Path) -> list[Page]:
    pages: list[Page] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value: object = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Invalid page on line {line_number}")
        if any(not isinstance(value.get(field), str) for field in ("id", "title", "url")):
            raise ValueError(f"Invalid page metadata on line {line_number}")
        raw_sections = value.get("sections")
        if not isinstance(raw_sections, list):
            raise ValueError(f"Invalid page sections on line {line_number}")
        sections: list[Section] = []
        for raw_section in raw_sections:
            if not isinstance(raw_section, dict):
                raise ValueError(f"Invalid section on line {line_number}")
            heading = raw_section.get("heading")
            blocks = raw_section.get("blocks")
            if (
                not isinstance(heading, str)
                or not isinstance(blocks, list)
                or not all(isinstance(block, str) for block in blocks)
            ):
                raise ValueError(f"Invalid section fields on line {line_number}")
            sections.append(Section(heading=heading, blocks=blocks))
        pages.append(Page(id=value["id"], title=value["title"], url=value["url"], sections=sections))
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Fetch live pages and replace the frozen page snapshot")
    args = parser.parse_args()
    corpus_dir = Path(__file__).resolve().parent.parent / "data" / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    pages_path = corpus_dir / "pages.jsonl"
    manifest_path = corpus_dir / "manifest.json"
    if args.refresh:
        pages: list[Page] = []
        sources = []
        fetched_at = datetime.now(UTC).isoformat()
        for source_id, url in SOURCES.items():
            html = fetch(url)
            page = extract(html, source_id, url)
            pages.append(page)
            sources.append(
                {
                    "id": source_id,
                    "url": url,
                    "title": page["title"],
                    "html_sha256": sha256(html),
                    "section_count": len(page["sections"]),
                }
            )
        write_jsonl(pages_path, pages)
        manifest_path.write_text(
            json.dumps(
                {
                    "fetched_at_utc": fetched_at,
                    "source": "GitHub Docs (Portuguese)",
                    "language": "pt-BR",
                    "source_license": "CC-BY-4.0; verify attribution before redistribution",
                    "sources": sources,
                    "extraction": "scripts/build_corpus.py",
                    "max_chunk_chars_target": MAX_CHARS,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        pages = read_pages(pages_path)
    chunks = make_chunks(pages)
    write_jsonl(corpus_dir / "chunks.jsonl", chunks)
    print(f"{len(pages)} pages, {len(chunks)} chunks")
    for page in pages:
        count = sum(chunk["source_id"] == page["id"] for chunk in chunks)
        print(f"  {page['id']}: {count} chunks")


if __name__ == "__main__":
    main()
