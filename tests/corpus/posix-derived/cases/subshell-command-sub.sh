x=outer
(x=inner; printf '%s\n' "$x")
printf '%s\n' "$x"
y=$(printf 'a\n\n\n')
printf '[%s]\n' "$y"
