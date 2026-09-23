case $SH in dash) exit ;; esac

set -u

echo plus /"${array[@]+foo}"/
echo plus colon /"${array[@]:+foo}"/
