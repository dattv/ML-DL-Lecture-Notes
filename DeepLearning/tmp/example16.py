from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow import data
import shutil
import math
from datetime import datetime
from tensorflow.python.feature_column import feature_column
from tensorflow.contrib.estimator.python import estimator as est


print(tf.__version__)

MODEL_NAME = 'class-model-01'

TRAIN_DATA_FILES_PATTERN = 'data/train-*.tfrecords'
VALID_DATA_FILES_PATTERN = 'data/valid-*.tfrecords'
TEST_DATA_FILES_PATTERN = 'data/test-*.tfrecords'

RESUME_TRAINING = False
PROCESS_FEATURES = True
EXTEND_FEATURE_COLUMNS = True
MULTI_THREADING = True

HEADER = ['key', 'x', 'y', 'alpha', 'beta', 'target']
HEADER_DEFAULTS = [[0], [0.0], [0.0], ['NA'], ['NA'], [0.0]]

NUMERIC_FEATURE_NAMES = ['x', 'y']

CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {'alpha': ['ax01', 'ax02'], 'beta': ['bx01', 'bx02']}
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys())

FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

TARGET_NAME = 'target'

TARGET_LABELS = ['positive', 'negative']

UNUSED_FEATURE_NAMES = list(set(HEADER) - set(FEATURE_NAMES) - {TARGET_NAME})

print("Header: {}".format(HEADER))
print("Numeric Features: {}".format(NUMERIC_FEATURE_NAMES))
print("Categorical Features: {}".format(CATEGORICAL_FEATURE_NAMES))
print("Target: {} - labels: {}".format(TARGET_NAME, TARGET_LABELS))
print("Unused Features: {}".format(UNUSED_FEATURE_NAMES))


def parse_tf_example(example_proto):
    feature_spec = {}

    for feature_name in NUMERIC_FEATURE_NAMES:
        feature_spec[feature_name] = tf.FixedLenFeature(shape=(), dtype=tf.float32)

    for feature_name in CATEGORICAL_FEATURE_NAMES:
        feature_spec[feature_name] = tf.FixedLenFeature(shape=(), dtype=tf.string)

    feature_spec[TARGET_NAME] = tf.FixedLenFeature(shape=(), dtype=tf.string)

    parsed_features = tf.parse_example(serialized=example_proto, features=feature_spec)

    target = parsed_features.pop(TARGET_NAME)

    return parsed_features, target


def process_features(features):
    features["x_2"] = tf.square(features['x'])
    features["y_2"] = tf.square(features['y'])
    features["xy"] = tf.multiply(features['x'], features['y'])  # features['x'] * features['y']
    features['dist_xy'] = tf.sqrt(tf.squared_difference(features['x'], features['y']))

    return features


def tfrecods_input_fn(files_name_pattern, mode=tf.estimator.ModeKeys.EVAL,
                      num_epochs=None,
                      batch_size=200):
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False

    print("")
    print("* data input_fn:")
    print("================")
    print("Input file(s): {}".format(files_name_pattern))
    print("Batch size: {}".format(batch_size))
    print("Epoch Count: {}".format(num_epochs))
    print("Mode: {}".format(mode))
    print("Shuffle: {}".format(shuffle))
    print("================")
    print("")

    file_names = tf.matching_files(files_name_pattern)
    dataset = data.TFRecordDataset(filenames=file_names)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda tf_example: parse_tf_example(tf_example))

    if PROCESS_FEATURES:
        dataset = dataset.map(lambda features, target: (process_features(features), target))

    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()

    features, target = iterator.get_next()
    return features, target


def extend_feature_columns(feature_columns, hparams):
    num_buckets = hparams.num_buckets
    embedding_size = hparams.embedding_size

    buckets = np.linspace(-3, 3, num_buckets).tolist()

    alpha_X_beta = tf.feature_column.crossed_column(
        [feature_columns['alpha'], feature_columns['beta']], 4)

    x_bucketized = tf.feature_column.bucketized_column(
        feature_columns['x'], boundaries=buckets)

    y_bucketized = tf.feature_column.bucketized_column(
        feature_columns['y'], boundaries=buckets)

    x_bucketized_X_y_bucketized = tf.feature_column.crossed_column(
        [x_bucketized, y_bucketized], num_buckets ** 2)

    x_bucketized_X_y_bucketized_embedded = tf.feature_column.embedding_column(
        x_bucketized_X_y_bucketized, dimension=embedding_size)

    feature_columns['alpha_X_beta'] = alpha_X_beta
    feature_columns['x_bucketized_X_y_bucketized'] = x_bucketized_X_y_bucketized
    feature_columns['x_bucketized_X_y_bucketized_embedded'] = x_bucketized_X_y_bucketized_embedded

    return feature_columns


def get_feature_columns(hparams):
    CONSTRUCTED_NUMERIC_FEATURES_NAMES = ['x_2', 'y_2', 'xy', 'dist_xy']
    all_numeric_feature_names = NUMERIC_FEATURE_NAMES.copy()

    if PROCESS_FEATURES:
        all_numeric_feature_names += CONSTRUCTED_NUMERIC_FEATURES_NAMES

    numeric_columns = {feature_name: tf.feature_column.numeric_column(feature_name)
                       for feature_name in all_numeric_feature_names}

    categorical_column_with_vocabulary = \
        {item[0]: tf.feature_column.categorical_column_with_vocabulary_list(item[0], item[1])
         for item in CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.items()}

    feature_columns = {}

    if numeric_columns is not None:
        feature_columns.update(numeric_columns)

    if categorical_column_with_vocabulary is not None:
        feature_columns.update(categorical_column_with_vocabulary)

    if EXTEND_FEATURE_COLUMNS:
        feature_columns = extend_feature_columns(feature_columns, hparams)

    return feature_columns


feature_columns = get_feature_columns(tf.contrib.training.HParams(num_buckets=5, embedding_size=3))
print("Feature Columns: {}".format(feature_columns))


def get_wide_deep_columns():
    feature_columns = list(get_feature_columns(hparams).values())

    dense_columns = list(
        filter(lambda column: isinstance(column, feature_column._NumericColumn) |
                              isinstance(column, feature_column._EmbeddingColumn),
               feature_columns
               )
    )

    categorical_columns = list(
        filter(lambda column: isinstance(column, feature_column._VocabularyListCategoricalColumn) |
                              isinstance(column, feature_column._BucketizedColumn),
               feature_columns)
    )

    sparse_columns = list(
        filter(lambda column: isinstance(column, feature_column._HashedCategoricalColumn) |
                              isinstance(column, feature_column._CrossedColumn),
               feature_columns)
    )

    indicator_columns = list(
        map(lambda column: tf.feature_column.indicator_column(column),
            categorical_columns)
    )

    deep_feature_columns = dense_columns + indicator_columns
    wide_feature_columns = categorical_columns + sparse_columns

    return wide_feature_columns, deep_feature_columns


def create_estimator(run_config, hparams, print_desc=False):
    wide_feature_columns, deep_feature_columns = get_wide_deep_columns()


    estimator = tf.estimator.DNNLinearCombinedClassifier(n_classes=len(TARGET_LABELS),
                                                         label_vocabulary=TARGET_LABELS,
                                                         dnn_feature_columns=deep_feature_columns,
                                                         linear_feature_columns=wide_feature_columns,
                                                         dnn_hidden_units=hparams.hidden_units,
                                                         dnn_optimizer=tf.train.AdamOptimizer(),
                                                         dnn_activation_fn=tf.nn.elu,
                                                         dnn_dropout=hparams.dropout_prob,
                                                         config=run_config
                                                         )

    if print_desc:
        print("")
        print("*Estimator Type:")
        print("================")
        print(type(estimator))
        print("")
        print("*deep columns:")
        print("==============")
        print(deep_feature_columns)
        print("")
        print("wide columns:")
        print("=============")
        print(wide_feature_columns)
        print("")

    return estimator


TRAIN_SIZE = 12000
NUM_EPOCHS = 1000
BATCH_SIZE = 500
NUM_EVAL = 10
TOTAL_STEPS = (TRAIN_SIZE / BATCH_SIZE) * NUM_EPOCHS
CHECKPOINT_STEPS = int((TRAIN_SIZE / BATCH_SIZE) * (NUM_EPOCHS / NUM_EVAL))

hparams = tf.contrib.training.HParams(
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    hidden_units=[16, 12, 8],
    num_buckets=6,
    embedding_size=3,
    max_steps=TOTAL_STEPS,
    dropout_prob=0.001)

model_dir = 'trained_models/{}'.format(MODEL_NAME)

run_config = tf.contrib.learn.RunConfig(
    save_checkpoints_steps=CHECKPOINT_STEPS,
    tf_random_seed=19830610,
    model_dir=model_dir
)

print(hparams)
print("Model Directory:", run_config.model_dir)
print("")
print("Dataset Size:", TRAIN_SIZE)
print("Batch Size:", BATCH_SIZE)
print("Steps per Epoch:", TRAIN_SIZE / BATCH_SIZE)
print("Total Steps:", TOTAL_STEPS)
print("Required Evaluation Steps:", NUM_EVAL)
print("That is 1 evaluation step after each", NUM_EPOCHS / NUM_EVAL, " epochs")
print("Save Checkpoint After", CHECKPOINT_STEPS, "steps")


def csv_serving_input_fn():
    SERVING_HEADER = ['x', 'y', 'alpha', 'beta']
    SERVING_HEADER_DEFAULTS = [[0.0], [0.0], ['NA'], ['NA']]

    rows_string_tensor = tf.placeholder(dtype=tf.string,
                                        shape=[None],
                                        name='csv_rows')

    receiver_tensor = {'csv_rows': rows_string_tensor}

    row_columns = tf.expand_dims(rows_string_tensor, -1)
    columns = tf.decode_csv(row_columns, record_defaults=SERVING_HEADER_DEFAULTS)
    features = dict(zip(SERVING_HEADER, columns))

    return tf.estimator.export.ServingInputReceiver(
        process_features(features), receiver_tensor)


train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: tfrecods_input_fn(
        TRAIN_DATA_FILES_PATTERN,
        mode=tf.estimator.ModeKeys.TRAIN,
        num_epochs=hparams.num_epochs,
        batch_size=hparams.batch_size
    ),
    max_steps=hparams.max_steps,
    hooks=None
)

eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: tfrecods_input_fn(
        VALID_DATA_FILES_PATTERN,
        mode=tf.estimator.ModeKeys.EVAL,
        num_epochs=1,
        batch_size=hparams.batch_size
    ),
    exporters=[tf.estimator.LatestExporter(
        name="predict",  # the name of the folder in which the model will be exported to under export
        serving_input_receiver_fn=csv_serving_input_fn,
        exports_to_keep=1,
        as_text=True)],
    steps=None,
    hooks=None
)

if not RESUME_TRAINING:
    print("Removing previous artifacts...")
    shutil.rmtree(model_dir, ignore_errors=True)
else:
    print("Resuming training...")

tf.logging.set_verbosity(tf.logging.INFO)

time_start = datetime.utcnow()
print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
print(".......................................")

estimator = create_estimator(run_config, hparams, True)

tf.estimator.train_and_evaluate(estimator=estimator,
                                train_spec=train_spec,
                                eval_spec=eval_spec
                                )

time_end = datetime.utcnow()
print(".......................................")
print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
print("")
time_elapsed = time_end - time_start
print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
