from calendar import c
import sys
import numpy as np
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

if 1:
    import cv2
    # sys.path.insert(0, '/opt/ros/melodic/lib/python2.7/dist-packages')
    # import cv2
    import queue
    import threading
    import time

class VideoCapture:

  def __init__(self, cap, name=None):
    # name = cap.name
    # print("starting video stream", name)
    if name == None:
      name = cap.name
    self.name = name
    self.q = queue.Queue() 
    self.cap = cap
    self.t = threading.Thread(target=self._reader)
    self.t.daemon = True
    self.enable = True
    self.t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while self.enable: 
      time.sleep(0.01)
      # cap.set(cv2.CAP_PROP_EXPOSURE, -10)
      ret, frame = self.cap.read()
      # cv2.imshow("frame", frame)
      # key = cv2.waitKey(1)
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get(timeout=5)

  def close(self):
    self.enable = False
    self.t.join()
    self.cap.close()


if __name__ == "__main__":
  import sys
  # resource = int(sys.argv[1])
  resource =  "/dev/video0"
  from franka_env.envs.capture.rs_capture import RSCapture
  side = RSCapture(name='side', serial_number='112222070712')
  cap = VideoCapture(side)
  while True:
    frame = cap.read()
    cv2.imshow('', frame)
    cv2.waitKey(1)