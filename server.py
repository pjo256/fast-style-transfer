import vimeo
import os
from flask import Flask, request

app = Flask(__name__)
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

@app.route('/stylize/<url>', methods=['GET'])
def stylize(url):
    style = request.args.get('style')
    if style not in styles:
        return "", 400
    
    output_file = './output/{}-stylized.mp4'.format(url)
    os.system(stylize_script.format(url, style, output_file))
    v.upload(output_file, data={'privacy': {'view':'nobody'}})
    
