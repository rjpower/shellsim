
# OSH had a bug here
`true`; echo $?
`false`; echo $?
$(true); echo $?
$(false); echo $?
echo ---

# OSH and others agree on these
eval true; echo $?
eval false; echo $?
`echo true`; echo $?
`echo false`; echo $?
