a=1
for ((  ;  ;  )); do
  if test $a = 4; then
    break
  fi
  echo $((a++))
done
