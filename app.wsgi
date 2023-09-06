activate_this = '/home/ubuntu/dent_app/venv/bin/activate'
with open(activate_this) as f:
    exec(f.read(), dict(__file__=activate_this))
import sys
import logging
logging.basicConfig(stream=sys.stderr) 
sys.path.insert(0,"/var/www/html/flaskapp/")
from app import app as application