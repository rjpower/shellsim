echo 1 && echo 2 || echo 3 && echo 4
echo --
false || echo A
false || false || echo B
false || false || echo C && echo D || echo E
