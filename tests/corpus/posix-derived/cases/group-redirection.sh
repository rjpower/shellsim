{
    printf 'first\n'
    printf 'second\n'
} > grouped
cat < grouped
printf 'outside\n'
