value=parent
printf 'child\n' | read value
printf 'after:%s\n' "$value"
