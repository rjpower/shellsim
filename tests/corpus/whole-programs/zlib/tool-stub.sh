#!/bin/sh
case "${0##*/}" in
  cc)
    output=a.out
    previous=
    for argument in "$@"; do
      if [ "$previous" = -o ]; then output=$argument; fi
      previous=$argument
    done
    printf '#!/bin/sh\nexit 0\n' >"$output"
    chmod +x "$output"
    ;;
esac
exit 0
