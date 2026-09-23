# What is the point of this?  ./configure scripts and debootstrap use it.
exec 3>&1
echo hi 1>&3
