x='a b  c'
set -- $x
printf '%s:%s:%s:%s\n' "$#" "$1" "$2" "$3"
set -- "$x"
printf '%s:[%s]\n' "$#" "$1"
