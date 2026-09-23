printf 'one\\ two\\\\three\n' | while read value; do
    printf 'default:<%s>\n' "$value"
done
printf 'one\\ two\\\\three\n' | while read -r value; do
    printf 'raw:<%s>\n' "$value"
done
