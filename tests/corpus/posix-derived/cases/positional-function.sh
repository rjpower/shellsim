set -- one 'two three'
show() { printf '[%s]\n' "$@"; }
show "$@"
shift
printf '%s:[%s]\n' "$#" "$1"
