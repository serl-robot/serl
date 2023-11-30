import numpy as np
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import cv2
from PIL import Image
from franka_env.camera.capture import Capture

class RSCapture(Capture):
  def get_device_serial_numbers(self):
    devices = rs.context().devices
    return [d.get_info(rs.camera_info.serial_number) for d in devices]
  
  def __init__(self, name, serial_number, dim = (640, 480), fps = 15, depth=False):
    self.name = name
    assert serial_number in self.get_device_serial_numbers()
    self.serial_number = serial_number
    self.depth = depth
    self.pipe = rs.pipeline()
    self.cfg = rs.config()
    self.cfg.enable_device(self.serial_number)
    self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
    if self.depth:
      self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)
    self.profile = self.pipe.start(self.cfg)

    if self.depth:
      # Getting the depth sensor's depth scale (see rs-align example for explanation)
      depth_sensor = self.profile.get_device().first_depth_sensor()
      depth_scale = depth_sensor.get_depth_scale()

      # Create an align object
      # rs.align allows us to perform alignment of depth frames to others frames
      # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    self.align = rs.align(align_to)

  def read(self):
    frames = self.pipe.wait_for_frames()
    aligned_frames = self.align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    if self.depth:
      depth_frame = aligned_frames.get_depth_frame()

    if color_frame.is_video_frame():
      image = np.asarray(color_frame.get_data())
      if self.depth and depth_frame.is_depth_frame():
        depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
        return True, np.concatenate((image, depth), axis=-1)
      else:
        return True, image
    else:
      return False, None

  def close(self):
    self.pipe.stop()
    self.cfg.disable_all_streams()

if __name__ == "__main__":
  cap = RSCapture(name='side_1', serial_number='128422270679', dim=(640, 480), depth=True)
  cap2 = RSCapture(name='side_2', serial_number='127122270146', dim=(640, 480), depth=True)
  cap3 = RSCapture(name='wrist_1', serial_number='127122270350', dim=(640, 480), depth=True)
  cap4 = RSCapture(name='wrist_2', serial_number='128422271851', dim=(640, 480), depth=True)

  while True:
    succes, img = cap.read()
    if succes:
      img = cv2.resize(img, (480, 640)[::-1])
      color_img = img[..., :3]
      depth_img = img[..., 3]
      depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
      img = np.hstack((color_img, depth_colormap))/255.
      cv2.imshow(cap.name, img)
      cv2.waitKey(1)
    succes2, img2 = cap2.read()
    if succes2:
      img = cv2.resize(img2, (480, 640)[::-1])
      color_img = img[..., :3]
      depth_img = img[..., 3]
      depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
      img = np.hstack((color_img, depth_colormap))/255.
      cv2.imshow(cap2.name, img)
      cv2.waitKey(1)
    succes3, img3 = cap3.read()
    if succes3:
      img = cv2.resize(img3, (480, 640)[::-1])
      color_img = img[..., :3]
      depth_img = img[..., 3]
      depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
      img = np.hstack((color_img, depth_colormap))/255.
      cv2.imshow(cap3.name, img)
      cv2.waitKey(1)
    succes4, img4 = cap4.read()
    if succes2:
      img = cv2.resize(img4, (480, 640)[::-1])
      color_img = img[..., :3]
      depth_img = img[..., 3]
      depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
      img = np.hstack((color_img, depth_colormap))/255.
      cv2.imshow(cap4.name, img)
      cv2.waitKey(1)