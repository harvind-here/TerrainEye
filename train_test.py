import os
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.applications import ResNet50V2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import shutil

tf.keras.mixed_precision.set_global_policy('mixed_float16')
# # Configure GPU memory growth
# physical_devices = tf.config.list_physical_devices('GPU')
# if physical_devices:
#     for device in physical_devices:
#         try:
#             tf.config.experimental.set_memory_growth(device, True)
#             tf.config.set_logical_device_configuration(device,[tf.config.LogicalDeviceConfiguration(memory_limit=3000)])
#         except RuntimeError as e:
#             print(e)

# 80/20
def split_dataset(source_dir, train_dir, test_dir, split_ratio=0.8):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for class_name in os.listdir(source_dir):
        if os.path.isdir(os.path.join(source_dir, class_name)):
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
            
            files = os.listdir(os.path.join(source_dir, class_name))
            np.random.shuffle(files)
            
            split_point = int(len(files) * split_ratio)
            
            for f in files[:split_point]:
                shutil.copy2(
                    os.path.join(source_dir, class_name, f),
                    os.path.join(train_dir, class_name, f)
                )
            for f in files[split_point:]:
                shutil.copy2(
                    os.path.join(source_dir, class_name, f),
                    os.path.join(test_dir, class_name, f)
                )

def plot_training_history(history, history_fine, save_dir='.'):
    # accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Phase 1 Training')
    plt.plot(history.history['val_accuracy'], label='Phase 1 Validation')
    plt.plot(range(len(history.history['accuracy']), 
             len(history.history['accuracy']) + len(history_fine.history['accuracy'])),
             history_fine.history['accuracy'], label='Phase 2 Training')
    plt.plot(range(len(history.history['val_accuracy']), 
             len(history.history['val_accuracy']) + len(history_fine.history['val_accuracy'])),
             history_fine.history['val_accuracy'], label='Phase 2 Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Phase 1 Training')
    plt.plot(history.history['val_loss'], label='Phase 1 Validation')
    plt.plot(range(len(history.history['loss']), 
             len(history.history['loss']) + len(history_fine.history['loss'])),
             history_fine.history['loss'], label='Phase 2 Training')
    plt.plot(range(len(history.history['val_loss']), 
             len(history.history['val_loss']) + len(history_fine.history['val_loss'])),
             history_fine.history['val_loss'], label='Phase 2 Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def evaluate_model(model_path, test_data_dir, img_height=256, img_width=256, batch_size=32):
    model = tf.keras.models.load_model(model_path)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20,20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()


def create_model(num_classes, img_height, img_width):
    base_model = ResNet50V2(
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

def validate_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except:
        return False

def clean_dataset(source_dir):
    corrupted = []
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path):
            for image in os.listdir(class_path):
                image_path = os.path.join(class_path, image)
                if not validate_image(image_path):
                    corrupted.append(image_path)
                    os.remove(image_path)
    return corrupted

def train_n_test():
    source_dir = r'C:\Users\harvi\Codebases\TerrainEye\dataset'
    train_dir = r'C:\Users\harvi\Codebases\TerrainEye\dataset_split\train'
    test_dir = r'C:\Users\harvi\Codebases\TerrainEye\dataset_split\test'
    split_dataset(source_dir, train_dir, test_dir)
    data_dir = train_dir

    batch_size = 32
    img_height = 256
    img_width = 256
    num_classes = 30

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    validation_generator = validation_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    print("Checking for corrupted images...")
    corrupted = clean_dataset(data_dir)
    if corrupted:
        print(f"Removed {len(corrupted)} corrupted images")
    else:
        print("No corrupted images found")

    model = create_model(num_classes, img_height, img_width)

    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    ]


    # Phase 1: to enable model to learn complex main features on final layers -- so to reduce overfit prblm.
    print("Phase 1: Training top layers...")
    history = model.fit(
        train_generator,
        epochs=30,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Phase 2: Fine-tuning by enabling base model and freeze initial layers as we want the pretrained feature extraction layers to be fixed.
    print("Phase 2: Fine-tuning...")
    base_model = model.layers[0]
    base_model.trainable = True

    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    model.save('TerrainEye.h5')
    plot_training_history(history, history_fine)
    # Save history data
    np.save('history.npy', history.history)
    np.save('history_fine.npy', history_fine.history)

    evaluate_model('TerrainEye.h5', test_dir)


if __name__ == "__main__":
    train_n_test()