from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(
        dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


import cv2

try:
    import pyscreenshot as ImageGrab
    from PIL import Image
    import pyautogui
    import time
except Exception as e:
    print(e)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    array = np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
    # print(array)
    return array

def screenshot(record_box_size=600):
    record_box_size = record_box_size // 2

    pos = pyautogui.position()
    mouse_x = pos[0]
    mouse_y = pos[1]
    left_top_x = mouse_x - record_box_size
    left_top_y = mouse_y - record_box_size
    right_bottom_x = mouse_x + record_box_size
    right_bottom_y = mouse_y + record_box_size

    frame = np.array(ImageGrab.grab(
        bbox=(left_top_x, left_top_y, right_bottom_x, right_bottom_y)))
    frame = cv2.resize(frame, (299, 299), interpolation=cv2.INTER_CUBIC)

    """
    image = Image.fromarray(frame)
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    """
    np_image_data = np.asarray(frame)
    np_image_data=cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
    np_final = np.expand_dims(np_image_data, axis=0)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("Beauty Detector", frame)
    if cv2.waitKey(110) & 0xff == 27:
        exit()

    return np_final

if __name__ == "__main__":
    file_name = "/home/yingshaoxo/Pictures/b.jpg"
    model_file = "../tensorflow/tf_files/girl_classify.pb" #"/home/yingshaoxo/Codes/tensorflow/tf_files/girl_classify.pb"
    label_file = "../tensorflow/tf_files/girl_classify_labels.txt" #"/home/yingshaoxo/Codes/tensorflow/tf_files/girl_classify_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "Placeholder"
    output_layer = "final_result"#"InceptionV3/Predictions/Reshape_1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.image:
        file_name = args.image
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer

    graph = load_graph(model_file)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    with tf.Session(graph=graph) as sess:
        while 1:
            print('-'*20)
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: screenshot(300)
            })

            results = np.squeeze(results)

            top_k = results.argsort()[-5:][::-1]
            labels = load_labels(label_file)
            for i in top_k:
                print(labels[i], float(results[i]))
