import queue
import threading
import time
# bufferless (Video) Capture
class Capture:

  def __init__(self, name):
    print("starting video stream", name)
    self.name = name
    
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True: 
      ret, frame = self.read_frame()
      if not ret:
        break 
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read_frame(self):
    raise NotImplementedError

  def read_recent(self):
    return self.q.get()

class MultiCapture:
  def __init__(self):
    self.streams = {}

  def add_stream(self, stream: Capture):
    name = stream.name
    if name in self.streams:
      return False
    self.streams[name] = stream
    return True

  def read_recent(self, name):
    return self.streams[name].read_recent()

  def read_all(self):
    frames = dict()
    import pdb; pdb.set_trace()
    for name in self.streams.keys():
      frames[name] = self.read_recent(name)
    return frames