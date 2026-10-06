#!/bin/sh

# Copy VNC config from Docker build into user space
cp -r /opt/run/.config $HOME/.config

# Now start supervisord daemon as root
dirname /opt/run/novnc/$EP_PATH | xargs mkdir -p
ln -s /opt/run/novnc/ /opt/run/novnc/$EP_PATH
/usr/bin/supervisord -c /opt/run/supervisord.conf

# Does not finish until supervisord exits
exit 0
