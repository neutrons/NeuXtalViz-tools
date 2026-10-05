#!/bin/sh

# Copy VNC config from Docker build into user space
cp -r /opt/run/.config $HOME/.config

# Now start supervisord daemon as root
dirname /root/novnc/$EP_PATH | xargs mkdir -p
ln -s /root/novnc/ /root/novnc/$EP_PATH
/usr/bin/supervisord

# Does not finish until supervisord exits
exit 0
