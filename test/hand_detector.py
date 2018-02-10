import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from protos import string_int_label_map_pb2
import os
import cv2

image_path = 'images/test.jpg'
# score threshold for showing bounding boxes.
score_thresh = 0.8
MODEL_NAME = 'hand_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')
detection_graph = tf.Graph()
NUM_CLASSES = 1


# Load a frozen infrerence graph into memory
def load_inference_graph():
    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess


# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)


# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
         detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


def _validate_label_map(label_map):
    """Checks if a label map is valid.

    Args:
      label_map: StringIntLabelMap to validate.

    Raises:
      ValueError: if label map is invalid.
    """
    for item in label_map.item:
        if item.id < 1:
            raise ValueError('Label map ids should be >= 1.')


def create_category_index(categories):
    """Creates dictionary of COCO compatible categories keyed by category id.

    Args:
      categories: a list of dicts, each of which has the following keys:
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name
          e.g., 'cat', 'dog', 'pizza'.

    Returns:
      category_index: a dict containing the same entries as categories, but keyed
        by the 'id' field of each category.
    """
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index


def convert_label_map_to_categories(label_map,
                                    max_num_classes,
                                    use_display_name=True):
    """Loads label map proto and returns categories list compatible with eval.

    This function loads a label map and returns a list of dicts, each of which
    has the following keys:
      'id': (required) an integer id uniquely identifying this category.
      'name': (required) string representing category name
        e.g., 'cat', 'dog', 'pizza'.
    We only allow class into the list if its id-label_id_offset is
    between 0 (inclusive) and max_num_classes (exclusive).
    If there are several items mapping to the same id in the label map,
    we will only keep the first one in the categories list.

    Args:
      label_map: a StringIntLabelMapProto or None.  If None, a default categories
        list is created with max_num_classes categories.
      max_num_classes: maximum number of (consecutive) label indices to include.
      use_display_name: (boolean) choose whether to load 'display_name' field
        as category name.  If False or if the display_name field does not exist,
        uses 'name' field as category names instead.
    Returns:
      categories: a list of dictionaries representing all possible categories.
    """
    categories = []
    list_of_ids_already_added = []
    if not label_map:
        label_id_offset = 1
        for class_id in range(max_num_classes):
            categories.append({
                'id': class_id + label_id_offset,
                'name': 'category_{}'.format(class_id + label_id_offset)
            })
        return categories
    for item in label_map.item:
        if not 0 < item.id <= max_num_classes:
            logging.info('Ignore item %d since it falls outside of requested '
                         'label range.', item.id)
            continue
        if use_display_name and item.HasField('display_name'):
            name = item.display_name
        else:
            name = item.name
        if item.id not in list_of_ids_already_added:
            list_of_ids_already_added.append(item.id)
            categories.append({'id': item.id, 'name': name})
    return categories


def load_labelmap(path):
    """Loads label map proto.

    Args:
      path: path to StringIntLabelMap proto text file.
    Returns:
      a StringIntLabelMapProto
    """
    with tf.gfile.GFile(path, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)
    _validate_label_map(label_map)
    return label_map


def get_label_map_dict(label_map_path):
    """Reads a label map and returns a dictionary of label names to id.

    Args:
      label_map_path: path to label_map.

    Returns:
      A dictionary mapping label names to id.
    """
    label_map = load_labelmap(label_map_path)
    label_map_dict = {}
    for item in label_map.item:
        label_map_dict[item.name] = item.id
    return label_map_dict


def main():
    #read image
    cap = cv2.imread(image_path)
    #resize image
    r = 500.0 / cap.shape[1]
    dim = (500, int(cap.shape[0] * r))
    cap = cv2.resize(cap, dim, interpolation=cv2.INTER_AREA)

    im_width, im_height = (cap.shape[1], cap.shape[0])
    # max number of hands we want to detect/track
    num_hands_detect = 1
    image_np = cap
    try:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    except:
        print("Error converting to RGB")
        quit(-1)

    # actual detection
    boxes, scores = detect_objects(
        image_np, detection_graph, sess)

    # draw bounding boxes
    draw_box_on_image(
        num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np)

    cv2.imshow('Hand Detection Result', cv2.cvtColor(
        image_np, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


if __name__ == '__main__':
    # load label map
    label_map = load_labelmap(PATH_TO_LABELS)
    categories = convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = create_category_index(categories)
    detection_graph, sess = load_inference_graph()
    # while using imread , it will return a numpy array.
    main()
