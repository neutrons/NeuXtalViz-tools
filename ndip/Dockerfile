FROM ubuntu:noble

ARG GITLAB_TOKEN

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

# Install Pixi
RUN curl -fsSL https://pixi.sh/install.sh | sh
ENV PATH="${HOME}/.pixi/bin:${PATH}"
ENV UV_COMPILE_BYTECODE=1

# Setup pixi.toml dependency files for /SNS/software scripts
RUN curl -fsSL --create-dirs -o /usr/local/pixi/scd-reduction-tools/pixi.toml \
    --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    "https://code.ornl.gov/api/v4/projects/18110/repository/files/prod%2Fdefault%2Fpixi.toml/raw?ref=main"
WORKDIR /usr/local/pixi/scd-reduction-tools
RUN pixi install
RUN curl -fsSL --create-dirs -o /usr/local/pixi/scd-reduction-tools_dev/pixi.toml \
    --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    "https://code.ornl.gov/api/v4/projects/18111/repository/files/prod%2Fdefault%2Fpixi.toml/raw?ref=main"
WORKDIR /usr/local/pixi/scd-reduction-tools_dev
RUN pixi install
RUN curl -fsSL --create-dirs -o /usr/local/pixi/shiver/pixi.toml \
    --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    "https://code.ornl.gov/api/v4/projects/18123/repository/files/prod%2Fdefault%2Fpixi.toml/raw?ref=main"
WORKDIR /usr/local/pixi/shiver
RUN pixi install

# Install FullProf
RUN curl -fsSL --create-dirs -o /opt/fullprof/fullprof.tgz https://www.ill.eu/sites/fullprof/downloads/FullProf_Suite_March2026_Linux64_ifort.tgz
WORKDIR /opt/fullprof
RUN tar -xvzf fullprof.tgz
RUN ln -s /opt/fullprof/fp_studio /opt/fullprof/fullprof
ENV FULLPROF=/opt/fullprof
ENV PATH="${FULLPROF}:${PATH}"

# Install VESTA
RUN curl -fsSL --create-dirs -o /opt/vesta/VESTA.tar.bz2 https://jp-minerals.org/vesta/archives/3.5.8/VESTA-gtk3.tar.bz2
WORKDIR /opt/vesta
RUN tar -xvjf VESTA.tar.bz2
ENV PATH="/opt/vesta/VESTA-gtk3:${PATH}"

# Build NeuXtalViz
COPY . /opt/NeuXtalViz
WORKDIR /opt/NeuXtalViz
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

RUN chmod -R og+rwX /opt /usr/local/pixi

EXPOSE 8080

ENTRYPOINT []
CMD ["/start.sh"]
