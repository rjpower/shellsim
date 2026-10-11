"""Parse and render fixed CommonMark input."""

from markdown_it import MarkdownIt

parser = MarkdownIt("commonmark")
tokens = parser.parse("hello `world`\n")
assert tokens[0].type == "paragraph_open"
assert any(token.type == "code_inline" and token.content == "world" for token in tokens[1].children)
assert parser.render("**hello**\n").strip() == "<p><strong>hello</strong></p>"
print("markdown-it-py: CommonMark tokens and HTML passed")
