set -eu
sqlite3 /work/cli.db < /work/transactions.sql > /work/cli-output
test "$(cat /work/cli-output)" = '1|committed'
test "$(sqlite3 /work/cli.db 'SELECT count(*) FROM items;')" = '1'
