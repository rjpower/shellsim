for value in a b; do printf 'for:%s\n' "$value"; done
n=0
while [ "$n" -lt 2 ]; do printf 'while:%s\n' "$n"; n=$((n + 1)); done
until [ "$n" -ge 3 ]; do n=$((n + 1)); done
case "$n" in 3) printf 'case:%s\n' "$n" ;; *) exit 9 ;; esac
