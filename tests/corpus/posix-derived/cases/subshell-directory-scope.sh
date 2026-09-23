mkdir -p child
start=$PWD
(cd child; changed=inside; printf 'inside:%s:%s\n' "$changed" "$PWD")
printf 'outside:%s:%s\n' "${changed-unset}" "$PWD"
test "$PWD" = "$start"
