import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Input,Lambda ,Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense,Dropout
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201,InceptionResNetV2
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152, VGG16,VGG19
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

def build_and_train_model_all(train_dataset, val_dataset, test_dataset, epochs=100,
                                           class_weight=None, factor=0.1, dropout_rate=0.2, 
                                           freeze_layers=40, min_lr=1e-8, patience=8, 
                                           restore_best_weights=True,architecture='InceptionResNet', lambda_value=0.001):

    if architecture == 'InceptionResNet':
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    elif architecture == 'DenseNet169':
        base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    elif architecture == 'DenseNet121':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    elif architecture == 'ResNet101':
        base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    elif architecture == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    elif architecture == 'EfficientNetB0':
        base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    elif architecture == 'EfficientNetB1':
        base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B1(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    elif architecture == 'EfficientNetB2':
        base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B2(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    elif architecture == 'EfficientNetB3':
        base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B3(weights='imagenet', include_top=False, input_shape=(512, 512, 3))    
    else:
        raise ValueError("Invalid architecture name. Choose from 'MobileNet', 'EfficientNetB0' to 'EfficientNetB7', 'Xception'.")

    

    # Add new layers for classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(lambda_value))(x)  # Adding L2 regularization
    x = BatchNormalization()(x)  # Adding Batch Normalization
    x = Dropout(dropout_rate)(x)  # Adding Dropout regularization
    predictions = Dense(2, activation='softmax')(x)  # Output layer with 2 units

    # Create the model
    model = models.Model(inputs=base_model.input, outputs=predictions)

    # Freeze the layers of the pre-trained model
    for layer in base_model.layers[:freeze_layers]:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=restore_best_weights)

    # Define ReduceLROnPlateau callback
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=2, min_lr=min_lr)

    # Train the model with callbacks
    history = model.fit(train_dataset,
                        epochs=epochs,
                        validation_data=val_dataset, 
                        class_weight=class_weight,
                        callbacks=[early_stopping, reduce_lr])

    return model, history


# Define directories
train_dir = ""
val_dir = ""
test_dir = ""

# Define image size
image_size = (512, 512)

# Define batch size
batch_size = 32

# Define the class names
class_names = ['class_1', 'class_2']

# Create train dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True,
)

# Create validation dataset
val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False,
)

# Create test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False,
)


# Extract labels from the datasets
train_labels = np.concatenate([y for _, y in train_dataset], axis=0)  # Extract labels from the train_dataset
# Convert class labels to integers if they are not already
train_labels_int = np.argmax(train_labels, axis=1)  # Assuming labels are one-hot encoded
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_labels_int), y=train_labels_int)
# Create a dictionary with integer keys for class weights
class_weight_dict = dict(zip(np.unique(train_labels_int), class_weights))
print(class_weight_dict)

model, history, acc = build_and_train_model_all(train_dataset, val_dataset, test_dataset, class_weight=class_weight_dict, factor=0.1, dropout_rate=0.4, freeze_layers=40, min_lr=1e-8, patience=8,restore_best_weights=False)
# Save the model to disk
save_model_dir = ""
model_name = ".h5"
save_model_path = os.path.join(save_model_dir, model_name)
model.save(save_model_path)