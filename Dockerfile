FROM jrottenberg/ffmpeg:3.1-alpine AS ffmpeg

FROM tensorflow/tensorflow:0.12.0-gpu AS tf
COPY --from=ffmpeg / /
RUN pip install -r requirements.txt
WORKDIR /src
COPY . .
RUN chmod +x stylize.sh
CMD ["/bin/bash", "-c", "./stylize.sh"]
