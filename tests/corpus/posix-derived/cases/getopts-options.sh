set -- -a -b value tail
OPTIND=1
while getopts 'ab:' option; do
    case "$option" in
        a) printf 'a\n' ;;
        b) printf 'b:%s\n' "$OPTARG" ;;
        ?) exit 8 ;;
    esac
done
shift $((OPTIND - 1))
printf 'rest:%s:<%s>\n' "$#" "$1"
