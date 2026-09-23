printf '  left right  \\tail\n' | while IFS= read -r line; do
    printf '<%s>\n' "$line"
done
printf 'a:b::c\n' | while IFS=: read first second rest; do
    printf '<%s><%s><%s>\n' "$first" "$second" "$rest"
done
