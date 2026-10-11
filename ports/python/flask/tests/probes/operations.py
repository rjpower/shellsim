"""Exercise local public behavior in the actual guest."""

from flask import Flask, jsonify

app = Flask(__name__)


@app.get("/double/<int:value>")
def double(value):
    return jsonify(value=2 * value)


client = app.test_client()
assert client.get("/double/7").json == {"value": 14}
assert client.get("/missing").status_code == 404
print("flask: guest probe passed")
