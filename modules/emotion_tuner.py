"""Tuning module
"""

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.v1.components import TunerFnResult

import keras_tuner as kt

LABEL_KEY = "label"
FEATURE_KEY = "text"

NUM_EPOCHS = 10
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 16

vectorize_layer = tf.keras.layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)

early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=3)


def transformed_name(key):
    """Renaming transformed features
    Args:
        key: feature key
    Returns:
        A string of transformed feature name
    """

    return key + "_xf"


def gzip_reader_fn(filenames):
    """Loads compressed data
    Args:
        filenames: input tfrecord file pattern.
    Returns:
        A TFRecordDataset
    """
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn(file_pattern,
             tf_transform_output,
             num_epochs,
             batch_size=64) -> tf.data.Dataset:
    """Generates features and labels for tuning/training.
    Args:
        file_pattern: input tfrecord file pattern.
        tf_transform_output: A TFTransformOutput.
        num_epochs: number of times to read the data. If None will loop through
        batch_size: representing the number of consecutive elements of 
        returned dataset to combine in a single batch
    Returns:
        A dataset that contains (features, indices) tuple where features
        is a dictionary of Tensors, and indices is a single Tensor of
        label indices.
    """

    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))
    return dataset


def model_builder(hparams: kt.HyperParameters, show_summary=True) -> tf.keras.Model:
    """Build a model based on given hyperparameters.
    Args:
        hparams: Holds HyperParameters for tuning.
        show_summary: Whether to show model summary.
    Returns:
        A keras model.
    """
    inputs = tf.keras.Input(
        shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = tf.keras.layers.Embedding(
        VOCAB_SIZE, EMBEDDING_DIM, name="embedding")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    for _ in range(hparams.get('num_layers')):
        x = tf.keras.layers.Dense(hparams.get(
            'num_units'), activation='relu')(x)

    outputs = tf.keras.layers.Dense(6, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hparams.get('learning_rate')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    if show_summary:
        model.summary()

    return model


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Tune the model based on given args.
    Args:
        fn_args: Holds args used to train the model as name/value pairs.
    Returns:
        A namedtuple contains the following:
        - A list of hyperparameter configs.
        - A dict of hyperparameters that is used to build the model.
        - A dict of the execution kwargs.
    """
    hp = kt.HyperParameters()
    hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    hp.Int('num_layers', 1, 3)
    hp.Int('num_units', 8, 64, step=8)

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    tuner = kt.RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=10,
        hyperparameters=hp,
        directory='output',
        project_name='emotion_classification_tuning'
    )

    train_set = input_fn(
        fn_args.train_files[0], tf_transform_output, NUM_EPOCHS)
    val_set = input_fn(fn_args.eval_files[0], tf_transform_output, NUM_EPOCHS)
    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
            for i in list(train_set)]])

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'callbacks': [early_stop_callback],
            'x': train_set,
            'validation_data': val_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )
