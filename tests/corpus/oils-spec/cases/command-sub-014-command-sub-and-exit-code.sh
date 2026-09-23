# A command resets the exit code, but an assignment doesn't.
echo $(echo x; exit 33)
echo $?
x=$(echo x; exit 33)
echo $?
