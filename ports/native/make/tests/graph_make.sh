set -eu
mkdir -p /tmp/make-graph
cd /tmp/make-graph
cat > Makefile <<'MAKE'
all: one two
	cat one two > result
one:
	printf 'one\n' > one
two:
	printf 'two\n' > two
fail:
	exit 7
MAKE
make -j2
test "$(cat result)" = "$(printf 'one\ntwo')"
if make fail; then
    exit 1
fi
