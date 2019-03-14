FROM jrottenberg/ffmpeg:3.1-alpine AS ffmpeg

FROM tiangolo/nginx-rtmp AS rtmp

FROM tensorflow/tensorflow:0.12.0-gpu
COPY --from=ffmpeg / /
COPY --from=rtmp / /
WORKDIR /src
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY nginx.conf /etc/nginx/nginx.conf
COPY . .

EXPOSE 1935
CMD ["nginx", "-g", "daemon off;"]
