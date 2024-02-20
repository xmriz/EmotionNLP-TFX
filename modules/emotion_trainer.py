"""Training module
"""

import os

import tensorflow as tf
import tensorflow_transform as tft
from keras.utils.vis_utils import plot_model
from tfx.components.trainer.fn_args_utils import FnArgs

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
    Returns:
        A keras model
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


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT.
    Then apply the model.
    Args:
        model: A Keras model to be served.
        tf_transform_output: A TFTransformOutput.
    Returns:
        A function that accepts a serialized tf.Example and returns the
        output of the model.
    """

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Parses raw tensors into a dict of tensors, then apply preprocessing
        and the model.
        Args:
            serialized_tf_examples: A batch of serialized tf.Example
            tensors.
        Returns:
            The outputs of the served model, a batch of probability
            tensors.
        """
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs):
    """Train the model based on given args.
    Args:
        fn_args: Holds args as name/value pairs.
    """

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch'
    )

    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
    mc = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir, monitor='val_accuracy', mode='max', verbose=1,
        save_best_only=True)

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)
    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
            for i in list(train_set)]])

    if fn_args.hyperparameters:
        hparams = kt.HyperParameters.from_config(fn_args.hyperparameters)
    else:
        hparams = {}

    model = model_builder(hparams)

    model.fit(x=train_set,
              validation_data=val_set,
              callbacks=[tensorboard_callback, es, mc],
              epochs=10,
              verbose=2)

    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'))
    }

    model.save(fn_args.serving_model_dir,
               save_format='tf', signatures=signatures)

    plot_model(
        model,
        to_file='images/model_plot.png',
        show_shapes=True,
        show_layer_names=True
    )
