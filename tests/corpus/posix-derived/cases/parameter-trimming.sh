path=/work/archive.tar.gz
printf 'length:%s\n' "${#path}"
printf 'prefix:%s:%s\n' "${path#*/}" "${path##*/}"
printf 'suffix:%s:%s\n' "${path%.*}" "${path%%.*}"
