"""Exercise a bounded public API in the actual guest."""

from bs4 import BeautifulSoup

soup = BeautifulSoup('<div><p class="item">A &amp; B</p><p>other</p></div>', "html.parser")
assert [p.get_text() for p in soup.select("div > p.item")] == ["A & B"]
assert soup.find("p", class_="item").parent.name == "div"
print("beautifulsoup4: guest public API probe passed")
