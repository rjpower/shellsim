false | true
printf 'pipe-right:%s\n' "$?"
true | false
printf 'pipe-fail:%s\n' "$?"
! false
printf 'negated:%s\n' "$?"
false && printf 'bad\n' || printf 'fallback\n'
