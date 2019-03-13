local VIDEO_URL = $0
local STYLE = $1
local OUTPUT_PATH = $2

INPUT_PATH = "./input/$VIDEO_URL-raw.mp4"

youtube-dl -o $INPUT_PATH $VIDEO_URL
python transform_video.py --in-path $INPUT_PATH \
  --checkpoint ./models/udnie.ckpt \
  --out-path $OUTPUT_PATH \
  --device /gpu:0 \
  --batch-size 1
