n=10
for ((a=1; a <= n ; a++))  # Double parentheses, and naked 'n'
do
  if test $a = 3; then
    continue
  fi
  if test $a = 6; then
    break
  fi
  echo $a
done
