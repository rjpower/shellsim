printf 'first\n' >out
printf 'second\n' >>out
printf 'problem\n' 2>err >&2
cat <out
cat err
