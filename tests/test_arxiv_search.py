"""Tests for the arXiv search tool."""


from src.tools.arxiv_search import _parse_arxiv_response


class TestParseArxivResponse:
    """Tests for parsing arXiv API XML responses."""

    def test_parse_valid_response(self) -> None:
        """Should parse a valid arXiv XML response."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Attention Is All You Need</title>
                <summary>We propose a new network architecture, the Transformer.</summary>
                <published>2026-05-15T00:00:00Z</published>
                <id>http://arxiv.org/abs/1706.03762v7</id>
                <author><name>Vaswani, Ashish</name></author>
            </entry>
        </feed>"""
        result = _parse_arxiv_response(xml)
        assert "Attention Is All You Need" in result
        assert "Vaswani, Ashish" in result
        assert "http://arxiv.org/abs/1706.03762v7" in result

    def test_parse_empty_response(self) -> None:
        """Should handle an empty feed."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>"""
        result = _parse_arxiv_response(xml)
        assert "No papers found" in result

    def test_parse_invalid_xml(self) -> None:
        """Should handle invalid XML gracefully."""
        result = _parse_arxiv_response("not xml at all")
        assert "Error" in result

    def test_parse_multiple_entries(self) -> None:
        """Should parse multiple entries and separate them."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Paper One</title>
                <summary>First paper abstract.</summary>
                <published>2026-05-15T00:00:00Z</published>
                <id>http://arxiv.org/abs/0001.0001</id>
                <author><name>Author A</name></author>
            </entry>
            <entry>
                <title>Paper Two</title>
                <summary>Second paper abstract.</summary>
                <published>2026-05-16T00:00:00Z</published>
                <id>http://arxiv.org/abs/0002.0002</id>
                <author><name>Author B</name></author>
            </entry>
        </feed>"""
        result = _parse_arxiv_response(xml)
        assert "Paper One" in result
        assert "Paper Two" in result
        assert "---" in result