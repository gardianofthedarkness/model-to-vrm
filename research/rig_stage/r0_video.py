"""Join PNG frames into an mp4 (OpenCV).  speed < 1 slows down by repeating nothing: it lowers the playback rate.
  E:\\splat\\.venv\\Scripts\\python.exe r0_video.py <frames dir> <out.mp4> [fps of the frames, default 60] [speed, default 1]"""
import sys, os, glob, cv2
D, OUT = sys.argv[1:3]; fps = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0; sp = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
fr = sorted(glob.glob(os.path.join(D, 'f*.png'))); h, w = cv2.imread(fr[0]).shape[:2]
for cc in ('avc1', 'mp4v'):
    vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*cc), fps * sp, (w, h))
    if vw.isOpened(): break
for f in fr: vw.write(cv2.imread(f))
vw.release(); print('wrote', OUT, len(fr), 'frames', cc, '%.0f fps' % (fps * sp))
