set -- "" ""
IFS=
echo "argv=${*-minus}"
echo "argv=${*+plus}"
echo "argv=${*:-minus}"
echo "argv=${*:+plus}"
