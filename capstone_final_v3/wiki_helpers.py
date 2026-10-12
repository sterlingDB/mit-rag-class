import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path


WIKIPEDIA_DIR = Path(__file__).parent / "Wikipedia"


class HtmlTextExtractor(HTMLParser):
    """Collect visible text from HTML and ignore scripts/styles."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg", "sup"}:
            self.skip_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag == self.skip_tag:
            self.skip_tag = None

    def handle_data(self, data: str) -> None:
        if self.skip_tag is None:
            text = data.strip()
            if text:
                self.parts.append(text)

    def text(self) -> str:
        text = unescape(" ".join(self.parts))
        return re.sub(r"\s+", " ", text).strip()


def article_html_only(html: str) -> str:
    """Keep the Wikipedia article area and skip most page navigation."""
    start_marker = 'id="mw-content-text"'
    start = html.find(start_marker)
    if start == -1:
        return html

    start = html.rfind("<div", 0, start)
    end = html.find("</main>", start)
    if end == -1:
        return html[start:]
    return html[start:end]


def html_to_text(html: str) -> str:
    """Turn a Wikipedia HTML file into plain text for retrieval."""
    article_html = article_html_only(html)
    parser = HtmlTextExtractor()
    parser.feed(article_html)
    return parser.text()


def read_wikipedia_article(filename: str) -> str:
    """Read one Wikipedia HTML file and return plain article text."""
    path = WIKIPEDIA_DIR / filename
    return html_to_text(path.read_text(encoding="utf-8", errors="replace"))

def load_docs_from_directory() -> list[dict[str, str]]:
    """Read every .html file from WIKIPEDIA_DIR into document dictionaries."""
    docs = []

    for path in sorted(WIKIPEDIA_DIR.glob("*.html")):
        docs.append({
            "id": path.stem,
            "text": html_to_text(path.read_text(encoding="utf-8", errors="replace")),
        })

    return docs
