"""Transform module
"""

import tensorflow as tf
from nltk.corpus import stopwords

LABEL_KEY = 'label'
FEATURE_KEY = 'text'

STOPWORDS = set(stopwords.words('english'))


def transformed_name(key):
    """Renaming transformed features
    Args:
        key: feature key
    Returns:
        A string of transformed feature name
    """
    return f'{key}_xf'


def preprocessing_fn(inputs):
    """Preprocess input features into transformed features
    Args:
        inputs: map from feature keys to raw features.
    Return:
        outputs: map from feature keys to transformed features.    
    """
    outputs = {}

    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(
        inputs[FEATURE_KEY])  # lowercase
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.regex_replace(
        outputs[transformed_name(FEATURE_KEY)], r"\b(" + "|".join(STOPWORDS) + ")\\W", "")
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.regex_replace(
        outputs[transformed_name(FEATURE_KEY)], r"['\"@#]", "")  # remove @, #, "
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.regex_replace(
        outputs[transformed_name(FEATURE_KEY)],
        r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "")  # remove urls
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.regex_replace(
        outputs[transformed_name(FEATURE_KEY)], r"[\n\t]", " ")  # remove newlines and tabs
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.regex_replace(
        outputs[transformed_name(FEATURE_KEY)], r"[^\w\s]", "")  # remove punctuation
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.regex_replace(
        outputs[transformed_name(FEATURE_KEY)], r"\s+", " ")  # remove extra spaces
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.strip(
        outputs[transformed_name(FEATURE_KEY)])  # remove leading and trailing spaces

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
