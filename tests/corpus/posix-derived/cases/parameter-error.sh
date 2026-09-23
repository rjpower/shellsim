( : "${missing:?required value}" ) 2>/dev/null
case $? in
    0) printf 'accepted-invalid-expansion\n' ;;
    *) printf 'rejected-invalid-expansion\n' ;;
esac
printf 'parent-alive\n'
