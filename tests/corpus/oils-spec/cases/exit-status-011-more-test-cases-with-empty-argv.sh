
true $(false)
echo status=$?

$(exit 42)
echo status=$?

$(exit 42) $(exit 43)
echo status=$?
