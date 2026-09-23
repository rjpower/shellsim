# What is the point of this?  ./configure scripts and debootstrap use it.
exec 3>&1
exec 4>&1
echo three 1>&3
echo four 1>&4
