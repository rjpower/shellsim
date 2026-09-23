for value in alpha beta-42 other; do
    case $value in
        a*) printf 'letter:%s\n' "$value" ;;
        *-[0-9][0-9]) printf 'number:%s\n' "$value" ;;
        *) printf 'other:%s\n' "$value" ;;
    esac
done
