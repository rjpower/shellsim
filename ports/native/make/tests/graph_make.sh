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

# GNU long-option parsing and bundled glob callbacks must remain available.
mkdir -p inputs
printf 'alpha\n' > inputs/alpha.txt
printf 'beta\n' > inputs/beta.txt
printf 'EXTRA = included\n' > extra-one.mk
cat > Glob.mk <<'MAKE'
include extra-*.mk
files := $(sort $(wildcard inputs/*.txt))
all:
	printf '%s\n' '$(files)' '$(EXTRA)' > glob-result
MAKE
make --file=Glob.mk --jobs=2 --no-print-directory all
test "$(cat glob-result)" = "$(printf 'inputs/alpha.txt inputs/beta.txt\nincluded')"
if make --shellsim-invalid-option; then
    exit 1
fi
make --file=Glob.mk --no-print-directory all
