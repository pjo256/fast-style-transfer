import vimeo
import os

styles = ["la_muse", "rain_princess", "scream", "udnie", "wave", "wreck"]
stylize_script = 'bash ./stylize.sh {} {} {}' 

client_id = os.environ['CLIENT_ID']
client_secret = os.environ['CLIENT_SECRET']
token = os.environ['TOKEN']

v = vimeo.VimeoClient(
    token=token,
    key=client_id,
    secret=client_secret
)


target_style = os.environ['STYLE']
if target_style not in styles:
    target_style = styles[0]

url = os.environ['URL']
output_file = './output/{}-stylized.mp4'.format(url)
os.system(stylize_script.format(url, target_style, output_file))
v.upload(output_file, data={'privacy': {'view':'nobody'}})
