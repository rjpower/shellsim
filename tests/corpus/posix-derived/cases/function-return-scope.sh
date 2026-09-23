set -- parent 'two words'
probe() {
    printf 'inside:%s:<%s>:<%s>\n' "$#" "$1" "$2"
    set -- changed
    return 7
}
probe child 'three words'
status=$?
printf 'outside:%s:%s:<%s>:<%s>\n' "$status" "$#" "$1" "$2"
