FROM jrottenberg/ffmpeg:3.1-alpine AS ffmpeg

FROM python:2.7.9 AS py
COPY --from=ffmpeg / /
COPY . .
RUN pip install -r requirements.txt
WORKDIR /src
RUN chmod +x stylize.sh
CMD ["/bin/bash", "-c", "./stylize.sh"]
