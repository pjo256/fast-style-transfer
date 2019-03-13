
FROM tensorflow/tensorflow:0.12.0-gpu
EXPOSE 7000
WORKDIR /src
ENV FLASK_APP /src/server/server.py
ENV DEBIAN_FRONTEND noninteractive
ENV PYCURL_SSL_LIBRARY=openssl
RUN add-apt-repository ppa:mc3man/trusty-media
RUN apt-get update
RUN apt-get install -y ffmpeg \
&& apt-get install -y libssl-dev \
&& apt-get install -y libcurl4-openssl-dev
RUN curl -Lo /usr/local/bin/youtube-dl https://yt-dl.org/downloads/latest/youtube-dl
RUN chmod a+rx /usr/local/bin/youtube-dl
RUN youtube-dl --version
COPY . .
RUN pip install -r requirements.txt
RUN chmod +x stylize.sh
