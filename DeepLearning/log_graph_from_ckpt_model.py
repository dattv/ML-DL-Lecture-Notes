import os
from remove import remove as rm
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True,
                    help="Input folder which is storing *.pb file")
parser.add_argument("-o", "--output", required=False,
                    help="Output folder which will store graph log file",
                    default="")
args = vars(parser.parse_args())


def export_graph_from_ckpt(input_folder=None, output_folder=None):
    if input_folder is None:
        print("{} does not exist".format(input_folder))
        return
    else:
        if os.path.isdir(input_folder) is False:
            print("{} does not exist".format(input_folder))
            return
        else:
            if output_folder is None:
                temp_folder = os.path.split(input_folder)
                tail_folder = temp_folder[1]
                head_folder = temp_folder[0]
                output_folder = os.path.join(head_folder, "log_" + tail_folder)

            tail_folder = os.path.split(input_folder)[1]
            g = tf.Graph()
            with g.as_default() as g:
                tf.train.import_meta_graph(os.path.join(input_folder, tail_folder + ".ckpt.meta"))

            with tf.Session(graph=g) as session:
                log_dir = output_folder
                rm.rm_dir(log_dir)
                file_writer = tf.summary.FileWriter(logdir=log_dir, graph=g)


if __name__ == '__main__':
    input_folder = args["input"]
    if args["output"] is "":
        output_folder = None

    export_graph_from_ckpt(input_folder, output_folder)
