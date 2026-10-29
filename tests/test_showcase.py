"""NeoShowcase 用の静的ページが単体で配信できることを確認する。"""

from __future__ import annotations

import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parent.parent
SHOWCASE = REPO_ROOT / "showcase"


class _AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.assets: list[str] = []
        self.ids: set[str] = set()
        self.anchors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if tag in {"img", "script"} and values.get("src"):
            self.assets.append(str(values["src"]))
        if tag == "link" and values.get("href"):
            self.assets.append(str(values["href"]))
        if tag == "a" and values.get("href", "").startswith("#"):
            self.anchors.append(str(values["href"])[1:])


class TestShowcase(unittest.TestCase):
    def setUp(self) -> None:
        source = (SHOWCASE / "index.html").read_text(encoding="utf-8")
        self.parser = _AssetParser()
        self.parser.feed(source)

    def test_local_assets_exist_inside_artifact(self) -> None:
        for asset in self.parser.assets:
            parsed = urlparse(asset)
            if parsed.scheme or asset.startswith("//"):
                continue
            path = (SHOWCASE / parsed.path).resolve()
            self.assertIn(SHOWCASE.resolve(), path.parents, asset)
            self.assertTrue(path.is_file(), asset)

    def test_page_anchors_have_targets(self) -> None:
        for anchor in self.parser.anchors:
            self.assertIn(anchor, self.parser.ids, anchor)


if __name__ == "__main__":
    unittest.main()
