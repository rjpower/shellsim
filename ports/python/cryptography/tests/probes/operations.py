"""Check real native hashes, authenticated encryption and signature errors."""

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

digest = hashes.Hash(hashes.SHA256())
digest.update(b"abc")
assert digest.finalize().hex() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
mac = hmac.HMAC(b"key", hashes.SHA256())
mac.update(b"message")
tag = mac.finalize()
check = hmac.HMAC(b"key", hashes.SHA256())
check.update(b"message")
check.verify(tag)
# RFC 5869, test case 1.
derived = HKDF(hashes.SHA256(), 42, bytes(range(13)), bytes(range(0xF0, 0xFA))).derive(b"\x0b" * 22)
assert derived.hex() == "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"

key = AESGCM(bytes(32))
nonce = bytes(12)
assert key.encrypt(nonce, b"", None).hex() == "530f8afbc74536b9a963b4f1c4cb738b"
encrypted = key.encrypt(nonce, b"payload", b"context")
assert key.decrypt(nonce, encrypted, b"context") == b"payload"
try:
    key.decrypt(nonce, encrypted, b"wrong context")
except InvalidTag:
    pass
else:
    raise AssertionError("invalid authentication tag accepted")

private = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
signature = private.sign(b"message")
private.public_key().verify(signature, b"message")
try:
    private.public_key().verify(signature, b"other")
except InvalidSignature:
    pass
else:
    raise AssertionError("invalid signature accepted")
print("cryptography: native hash, HMAC, HKDF, AES-GCM and Ed25519 passed")
