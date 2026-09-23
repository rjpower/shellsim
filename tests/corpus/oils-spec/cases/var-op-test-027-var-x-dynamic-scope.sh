f() { : "${hello:=x}"; echo $hello; }
f
echo hello=$hello

f() { hello=x; }
f
echo hello=$hello
