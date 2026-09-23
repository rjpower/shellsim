unset v
printf '%s:%s:%s:%s\n' "${v-default}" "${v:-default}" "${v+alt}" "${v:+alt}"
v=
printf '%s:%s:%s:%s\n' "${v-default}" "${v:-default}" "${v+alt}" "${v:+alt}"
v=value
printf '%s:%s:%s:%s\n' "${v-default}" "${v:-default}" "${v+alt}" "${v:+alt}"
unset v
printf '%s:%s\n' "${v:=assigned}" "$v"
