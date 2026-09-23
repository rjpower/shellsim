#!/bin/sh
set -eu

output=/work/output
while getopts 'o:' option; do
    case "$option" in
        o) output=$OPTARG ;;
        *) exit 2 ;;
    esac
done
shift "$((OPTIND - 1))"

if [ "$#" -ne 1 ]; then
    printf 'usage: build-index [-o directory] input\n' >&2
    exit 2
fi

mkdir -p "$output"
grep -v '^#' "$1" | sed '/^$/d' | sort -u > "$output/records"
cut -d: -f1 "$output/records" > "$output/names"
count=$(wc -l < "$output/records")
printf 'indexed %s records\n' "$count"
