emit() {
    printf 'out\n'
    printf 'err\n' >&2
}
emit >both 2>&1
emit 2>&1 >out-only
printf 'both:'
cat both
printf 'out-only:'
cat out-only
