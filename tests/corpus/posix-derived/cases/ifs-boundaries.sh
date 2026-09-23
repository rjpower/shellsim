value=':a::b:'
IFS=:
set -- $value
printf 'colon:%s' "$#"
for field in "$@"; do printf '<%s>' "$field"; done
printf '\n'
IFS=
set -- $value
printf 'empty:%s:<%s>\n' "$#" "$1"
