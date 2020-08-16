"""
Train our LSTM on extracted features.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
from extract_features import extract_features
import time
import os.path
import sys

def train(seq_length, saved_model=None, class_limit=None, image_shape=(224, 224, 3), load_to_memory=False, batch_size=32, nb_epoch=100):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join("Models", 'model_{epoch:03d}_{loss:.3f}_{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('Images', 'logs', "lstm"))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=10)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('Images', 'logs', "lstm" + '-' + 'training-' + \
        str(timestamp) + '.log'))

    # Get the data and process it.
    data = DataSet(
        seq_length=seq_length,
        class_limit=class_limit,
        image_shape=image_shape
    )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (data.data_length * 0.7) // batch_size

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory('train', "features")
        X_test, y_test = data.get_all_sequences_in_memory('test', "features")
    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'train', "features")
        val_generator = data.frame_generator(batch_size, 'test', "features")

    # Get the "lstm".
    rm = ResearchModels(len(data.classes), seq_length, saved_model)

    # Fit!
    if load_to_memory:
        # Use standard fit.
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            validation_data=val_generator,
            validation_steps=40)
    
    #save model
    # rm.model.save("model.h5")

def main():
    """These are the main training settings. Set each before running
    this file."""

    if (len(sys.argv) == 5):
        seq_length = int(sys.argv[1])
        class_limit = int(sys.argv[2])
        image_height = int(sys.argv[3])
        image_width = int(sys.argv[4])
    else:
        seq_length = 50
        class_limit = 2
        image_height = 120 # 120 # 240 # 480
        image_width = 160 # 160 # 320 # 640

    models_dir = "Models"
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    sequences_dir = os.path.join('Images', 'sequences')
    if not os.path.exists(sequences_dir):
        os.mkdir(sequences_dir)

    checkpoints_dir = os.path.join('Images', 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    # model can be only 'lstm'
    saved_model = None  # None or weights file
    load_to_memory = True # pre-load the sequences into memory
    batch_size = 32
    nb_epoch = 50
    image_shape = (image_height, image_width, 3)

    extract_features(seq_length=seq_length, class_limit=class_limit, image_shape=image_shape)
    train(seq_length, saved_model=saved_model, class_limit=class_limit, image_shape=image_shape, load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
