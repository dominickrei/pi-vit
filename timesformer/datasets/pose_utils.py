from scipy.ndimage.morphology import binary_dilation
from einops import rearrange

import numpy as np
import simplejson
import math

def json_2_keypoints(pose_path):
  '''
  Used for Smarthome. Get keypoint matrix given path to pose json

  ** Arguments **
  pose_path : str
    Path to {video_id}_pose3d.json

  ** Returns **
  keypoints : np.ndarray
    A matrix of 2d keypoints. (N, 13, 2)
  mask : np.ndarray
    A mask indicating frames containing keypoints
  '''
  filehandle = open(f'{pose_path}', 'rb')
  json = simplejson.load(filehandle)
  njts = json['njts']
#  K = json['K']
  num_frames = len(json['frames'])

  keypoints = np.zeros((num_frames, njts, 2)) - 1
  mask = np.zeros(num_frames)

  for frm_idx in range(num_frames):
    if (len(json['frames'][frm_idx]) >= 1) and (np.all(np.array(json['frames'][frm_idx][0]['pose2d']) != -1)): # frames containing no keypoints are represented as an empty list
      frm_pose_info = np.array(json['frames'][frm_idx][0]['pose2d'])
      frm_pose_info_p = np.reshape(frm_pose_info, (njts, 2), 'F')

      keypoints[frm_idx, :, :] = frm_pose_info_p
      mask[frm_idx] += 1

  return keypoints, mask, njts

def npy_to_keypoints(pose_path):
  '''
  Used for NTU60 and NTU120. Get keypoint matrix given path to a .skeleton.npy file.

  NOTE: NTU can have multiple bodies per video. At the moment we only return keypoints for a single body

  ** Arguments **
  pose_path : str
    Path to {video_id}.skeleton.npy

  ** Returns **
  keypoints : np.ndarray
    A matrix of 2d keypoints. (N, 25, 2) where N is the number of frames
  njts : int
    The number of joints
  '''
  np_skeleton = np.load(pose_path, allow_pickle=True).item()

  keypoints = np_skeleton['rgb_body0']
  njts = np_skeleton['njoints']

  return keypoints, njts

def keypoints_2_patch_idx(keypoints, patch_size, frame_width, frame_height, inflation=None):
  '''
  Convert a video's keypoint matrix -> patch mask matrix

  Mask matrix (M) is of size (num_frames, (frame_w // patch_size) * (frame_h // patch_size))
  M_(:, i) = 1 if a keypoint is contained in patch i, 0 otherwise
  
  ** Arguments **
  keypoints : np.ndarray
    Set of keypoints over relevant video frames
  patch_size : int
    The patch size of the TimeSformer
  frame_width : int
    Width of the video
  frame_height : int
    Height of the video

  ** Returns **
  mask : np.ndarray
    The patch mask matrix
  '''
  num_frames = keypoints.shape[0]
  num_patches_x = frame_width // patch_size
  num_patches_y = frame_height // patch_size

  patch_matrix = np.zeros( (num_frames, num_patches_x, num_patches_y) )

  for frm_idx, frm_kpts in enumerate(keypoints):
    for x, y in frm_kpts:
      if math.isnan(x) or math.isnan(y):
        continue

      x, y = int(x), int(y)

      # keypoint prediction may be outside of the frame, in this case ignore it
      if (x < 0) or (y < 0) or (x >= frame_width) or (y >= frame_height):
        continue
      else:
        patch_matrix[frm_idx, x // patch_size, y // patch_size] = 1

  if inflation is not None:
    kernel = np.ones((2*inflation + 1, 2*inflation + 1), dtype=bool) # seq = 2i + 1

    # probably a more efficient way to do this. figure this out if dataloading takes too long
    for i in range(patch_matrix.shape[0]):
      patch_matrix[i] = binary_dilation(patch_matrix[i], kernel, iterations=1).astype(int)

  patch_matrix = patch_matrix.reshape(num_frames * num_patches_x * num_patches_y)

  return patch_matrix

def keypoints_2_patch_joint_labels(keypoints, patch_size, frame_width, frame_height, njts):
  '''
  Convert a video's keypoint matrix -> matrix indicating which joint is in which patch

  Returned mask matrix (M) is of size (num_frames, (frame_w // patch_size), (frame_h // patch_size), njts)
  
  ** Arguments **
  keypoints : np.ndarray
    Set of keypoints over relevant video frames
  patch_size : int
    The patch size of the TimeSformer
  frame_width : int
    Width of the video
  frame_height : int
    Height of the video

  ** Returns **
  mask : np.ndarray
    The patch mask matrix
  '''
  num_frames = keypoints.shape[0]
  num_patches_x = frame_width // patch_size
  num_patches_y = frame_height // patch_size

  patch_matrix = np.zeros( (num_frames, num_patches_x, num_patches_y, njts) )

  for frm_idx, frm_kpts in enumerate(keypoints):
    for kpt_label, (x, y) in enumerate(frm_kpts):
      if math.isnan(x) or math.isnan(y):
        continue

      x, y = int(x), int(y)

      # keypoint prediction may be outside of the frame, in this case ignore it
      if (x < 0) or (y < 0) or (x >= frame_width) or (y >= frame_height):
        continue
      else:
        patch_matrix[frm_idx, x // patch_size, y // patch_size, kpt_label] = 1

  # Do not allow pose inflation if we are doing multi-label multi-class classification of joints
#   if inflation is not None:
#     kernel = np.ones((2*inflation + 1, 2*inflation + 1), dtype=bool) # seq = 2i + 1

#     # probably a more efficient way to do this. figure this out if dataloading takes too long
#     for i in range(patch_matrix.shape[0]):
#       patch_matrix[i] = binary_dilation(patch_matrix[i], kernel, iterations=1).astype(int)

  # Do not reshape if we are doing multi-label multi-class classification of joints
  # We want to use this to train an auxiliary loss
#   patch_matrix = patch_matrix.reshape(num_frames * num_patches_x * num_patches_y)

  patch_matrix = rearrange(patch_matrix, 't h w j -> (t h w) j', t=num_frames, w=num_patches_x, h=num_patches_y, j=njts)

  return patch_matrix
