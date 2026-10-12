"""Load a fixed CPython test CA and emit a TLS ClientHello through memory BIOs."""

import ssl

context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
context.load_verify_locations(cafile="/work/ca.pem")
assert context.cert_store_stats()["x509_ca"] == 1
assert len(context.get_ca_certs()) == 1
incoming = ssl.MemoryBIO()
outgoing = ssl.MemoryBIO()
connection = context.wrap_bio(incoming, outgoing, server_hostname="localhost")
try:
    connection.do_handshake()
except ssl.SSLWantReadError:
    pass
else:
    raise AssertionError("TLS handshake completed without a peer")
hello = outgoing.read()
assert hello[0] == 22 and len(hello) > 5
try:
    context.load_verify_locations(cadata="invalid certificate")
except ssl.SSLError:
    pass
else:
    raise AssertionError("invalid CA certificate accepted")
print("ssl context, CA loading and memory-BIO ClientHello passed")
