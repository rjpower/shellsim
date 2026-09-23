set -e
false && printf 'bad-and\n'
! false
if false; then printf 'bad-if\n'; fi
false | true
printf 'alive\n'
