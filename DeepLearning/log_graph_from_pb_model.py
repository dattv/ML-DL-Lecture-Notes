import os
import sys

from tensorflow.python.tools import import_pb_to_tensorboard
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from remove import remove as rm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True,
                    help="Input folder which is storing *.pb file")
parser.add_argument("-o", "--output", required=False,
                    help="Output folder which will store graph log file",
                    default="")
args = vars(parser.parse_args())


def export_graph_from_pb(input_folder=None, output_folder=None):
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

            rm.rm_dir(output_folder)
            # import_pb_to_tensorboard.import_to_tensorboard(input_folder, output_folder)

            GRAPH_PB_PATH = os.path.join(input_folder, os.path.split(input_folder)[1] + ".pb")
            with tf.Session() as session:
                with gfile.FastGFile(GRAPH_PB_PATH, "rb") as f:
                    data1 = f.read()
                    data = tf.compat.as_bytes(data1)
                    sm = saved_model_pb2.SavedModel()
                    sm.ParseFromString(data)
                    if len(sm.meta_graphs) > 1:
                        print("{} graphs detected, this tool have not support for saving multi-graphs yet".format(
                            len(sm.meta_graphs)))
                        sys.exit()
                    elif len(sm.meta_graphs) == 0:
                        sm = tf.GraphDef()
                        sm.ParseFromString(data1)
                        tf.import_graph_def(sm)
                    else:
                        tf.import_graph_def(sm.meta_graphs[0].graph_def)

                    if os.path.isdir(output_folder) is False:
                        os.mkdir(output_folder)

                    writer = tf.summary.FileWriter(output_folder)
                    writer.add_graph(session.graph)


if __name__ == '__main__':
    input_folder = args["input"]
    if args["output"] is "":
        output_folder = None

    export_graph_from_pb(input_folder, output_folder)
