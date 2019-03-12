source activate fast-style-transfer
echo $(which python)
python transform_video.py --in-path ./video/sample.mp4 \
  --checkpoint ./models/udnie.ckpt \
  --out-path ./output/stylized.mp4 \
  --device /gpu:0 \
  --batch-size 4
