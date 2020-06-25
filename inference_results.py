import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import zipfile
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as et
import xml

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


#load_model

model_dir = ("./Damage_only/inference_model/frozen_inference_graph.pb")

detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.gfile.GFile(model_dir, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Add a wrapper function to call the model, and cleanup the outputs:

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.compat.v1.Session(config=config) as sess:
      # Get handles to input and output tensors
      ops = tf.compat.v1.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './pascal_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)

# test image location
image_path = './VOC/VOC2007/JPEGImages'
gt_path = './New_Annotations/Damage_only'
image_sample = 'C:/Users/MoMoJo/Desktop/Faster_resNet/VOC/VOC2007/JPEGImages/chrisear001.jpg'

test_file_path =('./VOC/VOC2007/ImageSets/Main/test.txt')
test_files = pd.read_csv(test_file_path, sep=" ", header=None)
output = pd.DataFrame(columns=('file','class', 'score'), index = None)

"""multiple images"""
for _name in tqdm(test_files[0]):
    img=Image.open(os.path.join(image_path,_name+'.jpg'))
    (im_width, im_height) = img.size
    # convert to tensor
    img_np = (
        np.array(img.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8))
    img_np_expanded = np.expand_dims(img_np, axis=0)
    #evalaute
    output_dict = run_inference_for_single_image(img_np, detection_graph)

    output=output.append(
        pd.DataFrame({'file':[_name],
        'class':[output_dict['detection_classes'][0]],
        'score':[output_dict['detection_scores'][0]]}),ignore_index=True)
output.to_csv('SSD_Type_only_resNet_results.csv')
""""""
img=Image.open(os.path.join(image_path,test_files[0][250]+'.jpg'))
#14'haiear019'
#251
tree = et.parse(os.path.join(gt_path,test_files[0][250]+'.xml'))
root = tree.getroot()
gt_class = root[6][0].text
gt_left = int(root[6][4][0].text)
gt_top = int(root[6][4][1].text)
gt_right = int(root[6][4][2].text)
gt_bottom = int(root[6][4][3].text)

(im_width, im_height) = img.size
# convert to tensor
img_np = (
    np.array(img.getdata()).reshape(
    (im_height, im_width, 3)).astype(np.uint8))
#img_np_expanded = np.expand_dims(img_np, axis=0)
#evalaute
output_dict = run_inference_for_single_image(img_np, detection_graph)

"""vis_util.visualize_boxes_and_labels_on_image_array(
  img_np,
  output_dict['detection_boxes'],
  output_dict['detection_classes'],
  output_dict['detection_scores'],
  category_index,
  instance_masks=output_dict.get('detection_masks'),
  use_normalized_coordinates=True, line_thickness=14)"""


image_pil = Image.fromarray(np.uint8(img_np.copy())).convert('RGB')
draw = ImageDraw.Draw(image_pil)
im_width, im_height = image_pil.size

(ymin, xmin, ymax, xmax) = output_dict['detection_boxes'][0]
(left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                            ymin * im_height, ymax * im_height)

draw.line([(left, top), (left, bottom), (right, bottom),(right, top), (left, top)],
    width=6, fill='black')

_x = gt_left
x_ = gt_right
_y = gt_top
y_ = gt_bottom
for x in range(_x, x_, 100):
    draw.line([(x, _y), (x + 50, _y)], width=6, fill='black')
    draw.line([(x, y_), (x + 50, y_)], width=6, fill='black')

for y in range(_y, y_, 100):
    draw.line([(_x, y), (_x, y + 50)], width=6, fill='black')
    draw.line([(x_, y), (x_, y + 50)], width=6, fill='black')

gt_str_list = 'GroundTruth Class:%s'%gt_class
display_str_list = 'Class:%s, Score: %f%%'%(output_dict['detection_classes'][0],
    output_dict['detection_scores'][0]*100) #
font = ImageFont.truetype('Times New Roman Bold.ttf', 18)

text_height = 18
display_str_widths = [font.getsize(ds)[0] for ds in display_str_list]
gt_str_widths = [font.getsize(ds)[0] for ds in gt_str_list]

total_display_str_width = (1 + 2 * 0.05) * sum(display_str_widths)
total_gt_str_width = (1 + 2 * 0.05) * sum(gt_str_widths)

margin = np.ceil(0.05 * text_height)

"""gt box label"""
if gt_top > text_height:
  text_bottom = gt_top-4
else:
  text_bottom = gt_bottom + text_height-4

draw.rectangle(
   [(gt_left, text_bottom - text_height - 2 * margin), (gt_left +
                                                     total_gt_str_width,
                                                     text_bottom)],
   fill='white')
draw.text(
  (gt_left + margin, text_bottom - text_height - margin),
  gt_str_list, fill='black', font=font,dpi=4800)

"""est box lable"""
if top > text_height:
  text_bottom = top-4
else:
  text_bottom = bottom + text_height-4

draw.rectangle(
   [(left, text_bottom - text_height - 2 * margin), (left +
                                                     total_display_str_width,
                                                     text_bottom)],
   fill='white')
draw.text(
  (left + margin, text_bottom - text_height - margin),
  display_str_list, fill='black', font=font, dpi = 4800)

""""""

image_resize = image_pil.resize((792,612))
image_resize.save('./moderate_DL_1.pdf')
