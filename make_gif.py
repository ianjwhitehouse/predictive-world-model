# Imports
from PIL import Image
import tensorflow as tf
import sys

# Load image
img_name = int(sys.argv[1])
frames = tf.io.parse_tensor(tf.io.read_file("runs/%d/frames.proto_tensor" % img_name), tf.uint8).numpy()
frames = [Image.fromarray(frame) for frame in frames]

# Make gif
frames[0].save(open("gifs/%d.gif" % img_name, "wb+"), save_all=True, append_images=frames[1:])
