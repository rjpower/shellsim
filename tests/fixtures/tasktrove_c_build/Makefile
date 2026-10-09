CC ?= cc
CFLAGS ?= -O2

all: solver

solver: brackets.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDFLAGS)
	chmod +x $@

clean:
	rm -f solver

.PHONY: all clean
