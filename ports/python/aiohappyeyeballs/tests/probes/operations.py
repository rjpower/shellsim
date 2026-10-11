"""Exercise offline address conversion and family interleaving."""

import socket

from aiohappyeyeballs import addr_to_addr_infos, pop_addr_infos_interleave, remove_addr_infos

ipv4 = addr_to_addr_infos(("127.0.0.1", 80))
ipv6 = addr_to_addr_infos(("::1", 80))
assert ipv4[0][0] == socket.AF_INET
assert ipv6[0][0] == socket.AF_INET6
assert ipv6[0][-1] == ("::1", 80, 0, 0)
assert addr_to_addr_infos(None) is None
working = ipv4 + ipv6 + ipv4
pop_addr_infos_interleave(working, 1)
assert working == ipv4
remove_addr_infos(working, ("127.0.0.1", 80))
assert working == []
try:
    remove_addr_infos(working, ("127.0.0.1", 80))
except ValueError:
    pass
else:
    raise AssertionError("missing address was accepted")
print("aiohappyeyeballs: offline address operations passed")
