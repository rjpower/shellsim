all: codec.wasm
codec.wasm: main.o codec.o
	cc -o $@ main.o codec.o -L/opt/zlib/lib -lz
	chmod +x $@
%.o: %.c
	cc -I/opt/zlib/include -c $< -o $@
