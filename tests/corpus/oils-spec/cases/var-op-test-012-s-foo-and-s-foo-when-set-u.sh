set -u
v=v
echo v=${v:+foo}
echo v=${v+foo}
unset v
echo v=${v:+foo}
echo v=${v+foo}
