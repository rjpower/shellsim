. "$1"
__shellsim_status=$?
printf '%s\n' "$__shellsim_status"
case "$__shellsim_status" in
  0|1) exit 0 ;;
  *) exit 2 ;;
esac
