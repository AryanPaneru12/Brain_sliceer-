from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPool2D,
    UpSampling2D,
)

from brain_tumor_pipeline.metrics import (
    dice_coefficient,
    focal_tversky,
    iou_coefficient,
    tversky,
)


def build_resnet50_classifier(
    input_shape: tuple[int, int, int] = (256, 256, 3),
    weights: str | None = "imagenet",
) -> Model:
    inputs = Input(shape=input_shape, name="image")
    backbone = ResNet50(
        weights=weights,
        include_top=False,
        input_shape=input_shape,
        name="resnet50_backbone",
    )
    backbone.trainable = False
    x = backbone(inputs, training=False)
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = Dense(256, activation="relu", name="classifier_dense_1")(x)
    x = Dropout(0.3, name="classifier_dropout_1")(x)
    x = Dense(256, activation="relu", name="classifier_dense_2")(x)
    x = Dropout(0.3, name="classifier_dropout_2")(x)
    outputs = Dense(2, activation="softmax", name="tumor_probability")(x)
    return Model(inputs=inputs, outputs=outputs, name="resnet50_tumor_classifier")


def compile_classifier(model: Model, learning_rate: float = 1e-4) -> Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def unfreeze_resnet_top_layers(
    model: Model,
    n_layers: int = 75,
    keep_batch_norm_frozen: bool = True,
) -> Model:
    backbone = model.get_layer("resnet50_backbone")
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers[-n_layers:]:
        if keep_batch_norm_frozen and isinstance(layer, BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
    return model


def _resblock(x, filters: int):
    shortcut = x
    x = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)
    shortcut = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), kernel_initializer="he_normal")(
        shortcut
    )
    shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    return Activation("relu")(x)


def _upsample_concat(x, skip):
    x = UpSampling2D((2, 2))(x)
    return Concatenate()([x, skip])


def build_resunet(input_shape: tuple[int, int, int] = (256, 256, 3)) -> Model:
    inputs = Input(input_shape, name="image")

    conv1 = Conv2D(16, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = _resblock(pool1, 32)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    conv3 = _resblock(pool2, 64)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    conv4 = _resblock(pool3, 128)
    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)
    conv5 = _resblock(pool4, 256)

    up1 = _upsample_concat(conv5, conv4)
    up1 = _resblock(up1, 128)
    up2 = _upsample_concat(up1, conv3)
    up2 = _resblock(up2, 64)
    up3 = _upsample_concat(up2, conv2)
    up3 = _resblock(up3, 32)
    up4 = _upsample_concat(up3, conv1)
    up4 = _resblock(up4, 16)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", name="mask")(up4)
    return Model(inputs=inputs, outputs=outputs, name="resunet_segmenter")


def compile_segmenter(model: Model, learning_rate: float = 1e-4) -> Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=focal_tversky,
        metrics=[tversky, dice_coefficient, iou_coefficient],
    )
    return model
