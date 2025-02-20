# import pydot_ng as pydot
# pydot.find_graphviz = lambda: True
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers

def create_model(num_classes, img_height, img_width):
    base_model = keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, 3))

    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

plot_model(
    create_model(30, 256, 256),
    to_file='model_architecture.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB'  # 'TB' for vertical plot; 'LR' for horizontal plot
)