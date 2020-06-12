## Requirements
<code> pip install -r requirements.txt </code>
## Usage

### To use webcam stream (Default)
<code>python detect.py</code>

### To use any video at a certain path
<code>python detect.py --path='path-to-file.mp4'</code>

### To use with a video stream from an IP Camera
```
python3 test.py --video rtsp://admin:admin@192.168.0.100/1
```

## Haar Cascades Used
- Full Body
- Upper Body
- Side Profile
