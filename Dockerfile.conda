FROM jrottenberg/ffmpeg:3.1-alpine AS ffmpeg

FROM continuumio/miniconda3 AS conda
COPY --from=ffmpeg / /
WORKDIR /src
COPY environment.yml .
RUN conda env create -f environment.yml
COPY . .
RUN chmod +x stylize.sh
CMD ["/bin/bash", "-c", "./stylize.sh"]
