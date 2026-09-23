trap 'printf "caught:USR1\n"' USR1
kill -USR1 $$
printf 'after-signal\n'
trap 'printf "on-exit:%s\n" "$?"' 0
