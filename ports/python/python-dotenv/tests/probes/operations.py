"""Parse mounted dotenv data and respect existing guest environment values."""

import os

from dotenv import dotenv_values, load_dotenv

source = "/work/config.env"
assert dotenv_values(source) == {
    "GUEST_NAME": "mounted guest",
    "GUEST_QUOTED": "hash # and equals =",
    "GUEST_GREETING": "hello mounted guest",
    "GUEST_EMPTY": "",
    "GUEST_BARE": None,
}
os.environ["GUEST_NAME"] = "environment"
assert load_dotenv(source, override=False)
assert os.environ["GUEST_NAME"] == "environment"
assert os.environ["GUEST_GREETING"] == "hello environment"
assert "GUEST_BARE" not in os.environ
assert load_dotenv(source, override=True)
assert os.environ["GUEST_NAME"] == "mounted guest"
assert os.environ["GUEST_GREETING"] == "hello mounted guest"

print("python-dotenv: mounted values passed")
