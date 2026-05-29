#!/bin/sh

# This username should be passed from the Pulsar laucher
if [ $# -lt 1 ]; then
    echo "usage: $0 username"
    exit 1
fi

USER=$1
GROUPS=$(id -G | cut -d' ' -f2-)

# Create relevant user groups within the container
INDEX=0
for group in $GROUPS; do
    groupadd -g $group "host_group_$INDEX"
    INDEX=$((INDEX+1))
done

# Create a user that is mapped to the unix user id
UID=`ldapsearch -LLL -x -b "dc=xcams,dc=ornl,dc=gov" -H ldaps://ldapx.ornl.gov/ "uid=$USER" uidNumber | sed -ne 's/^uidNumber: //p'`
if [ -z "$UID" ]; then
    echo "Unable to find UID for user \"$USER\""
    exit 1
fi
useradd --uid $UID novnc_user
usermod -aG $(echo $GROUPS | sed 's/\s\+/,/g') novnc_user
groupmod -g $UID novnc_user
chown -R novnc_user /home/novnc_user

# Now start supervisord daemon as root
dirname /root/novnc/$EP_PATH | xargs mkdir -p
ln -s /root/novnc/ /root/novnc/$EP_PATH
/usr/bin/supervisord

# Does not finish until supervisord exits
exit 0
