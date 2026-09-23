i=1
for ((  ; ;  i++ )); do
  if test $i = 4; then
    break
  fi
  echo $i
done
