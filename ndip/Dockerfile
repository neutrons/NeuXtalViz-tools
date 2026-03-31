FROM ubuntu:noble

ENV HOME /home/novnc_user
ENV DEBIAN_FRONTEND noninteractive
ENV LC_ALL C.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV USER=root

RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get -y install libegl1 libgl1 libglx-mesa0 libegl-mesa0 libglapi-mesa libosmesa6 libasound2t64 python3 libnss3 pip \
    xvfb x11vnc xdotool wget tar supervisor net-tools openbox tint2 gnupg2 ldap-utils libxcb-xinerama0 '^libxcb.*-dev' libx11-xcb-dev \
    libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev libsystemd-dev pkg-config xinit libxcvt-dev curl && \
    apt-get -y full-upgrade && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ndip/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
ENV DISPLAY :1

WORKDIR $HOME/
RUN wget -O - https://github.com/novnc/noVNC/archive/v1.4.0.tar.gz | tar -xzv -C /root/ && mv /root/noVNC-1.4.0 /root/novnc && ln -s /root/novnc/vnc_lite.html /root/novnc/index.html && \
    wget -O - https://github.com/novnc/websockify/archive/v0.11.0.tar.gz | tar -xzv -C /root/ && mv /root/websockify-0.11.0 /root/novnc/utils/websockify && \
    wget https://sourceforge.net/projects/tigervnc/files/stable/1.15.0/ubuntu-24.04LTS/amd64/tigervncserver_1.15.0-1ubuntu1_amd64.deb && dpkg -i ./tigervncserver_1.15.0-1ubuntu1_amd64.deb

# Build NeuXtalViz
COPY . /opt/NeuXtalViz
WORKDIR /opt/NeuXtalViz
RUN curl -fsSL https://pixi.sh/install.sh | sh
ENV PATH="${HOME}/.pixi/bin:${PATH}"
ENV UV_COMPILE_BYTECODE=1
RUN pixi init --import environment.yml
RUN pixi add pyqt qdarkstyle qtawesome
RUN pixi install
SHELL [ "pixi", "run" ]

# Configure openbox to show a custom menu on a noVNC client event.
COPY ndip/openbox/autostart /etc/xdg/openbox/autostart
COPY ndip/openbox/rc.xml /etc/xdg/openbox/rc.xml
COPY ndip/openbox/menu.xml /etc/xdg/openbox/menu.xml
COPY ndip/openbox/menu.xml /var/lib/openbox/debian-menu.xml
COPY ndip/openbox/tint2rc /etc/xdg/tint2/tint2rc

COPY ndip/start.sh /start.sh
RUN chmod +x /start.sh
COPY ndip/vnc_lite.html /root/novnc/vnc_lite.html
RUN mkdir -p $HOME/.config/tigervnc
COPY ndip/tigervnc.conf $HOME/.config/tigervnc/config
COPY ndip/user_command.sh $HOME/user_command.sh
RUN chmod +x $HOME/user_command.sh

RUN chmod -R og+rwX /opt

EXPOSE 8080

ENTRYPOINT []
CMD ["/start.sh"]
