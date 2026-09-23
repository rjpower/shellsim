set -- 'a b' '' c
printf 'quoted-star:'
for field in "$*"; do printf '<%s>' "$field"; done
printf '\nquoted-at:'
for field in "$@"; do printf '<%s>' "$field"; done
printf '\nunquoted-star:'
for field in $*; do printf '<%s>' "$field"; done
printf '\n'
