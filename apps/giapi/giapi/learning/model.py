import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from gicommon.models.learning import DLModel, KerasModelEnum
from giapi.preprocessing.dataset import DatasetManager
from giapi.config import CONFIG
from giapi.logging import logger


class DLModelManager:
    def __init__(self, dl_model: DLModel):
        self._dl_model = dl_model
        self._image_shape = (self._dl_model.dataset.image_size, self._dl_model.dataset.image_size, 3)

    @staticmethod
    def get_path(dl_model: DLModel, sub_folder_name):
        formatted_date = dl_model.created_at.strftime('%Y-%m-%d-%H-%M-%S')
        return os.path.join(CONFIG.data_path, sub_folder_name, f'{formatted_date}_{str(dl_model.id)}')

    @staticmethod
    def get_save_path(dl_model: DLModel):
        return f"{DLModelManager.get_path(dl_model, 'models')}.h5"

    @staticmethod
    def get_logs_path(dl_model: DLModel):
        return DLModelManager.get_path(dl_model, 'logs')

    @staticmethod
    def scale_image(image):
        return (image / 255.0 - 0.5) * 2

    def get_generators(self):
        aug_datagen = ImageDataGenerator(
            preprocessing_function=DLModelManager.scale_image,
            rotation_range=90,
            zoom_range=[0.5, 2.0],
            horizontal_flip=True,
            # vertical_flip=True,
            # width_shift_range=0.2,
            # brightness_range=[0.2, 0.8]
        )
        std_datagen = ImageDataGenerator(
            preprocessing_function=DLModelManager.scale_image
        )
        return aug_datagen, std_datagen

    def get_model(self, compile_model=False):
        # Define the base model
        weights = 'imagenet' if self._dl_model.use_imagenet_weights else None

        available_base_models = __import__('tensorflow.keras.applications', fromlist=KerasModelEnum.to_list())
        base_model_class = getattr(available_base_models, self._dl_model.base_model_name)
        base_model = base_model_class(
            weights=weights,
            input_shape=self._image_shape,
            include_top=False,
        )
        base_model.trainable = self._dl_model.base_model_trainable

        # Define the classifier
        inputs = Input(shape=self._image_shape)
        x = base_model(inputs)
        x = Flatten()(x)
        if self._dl_model.include_relu_dense:
            x = Dense(self._dl_model.relu_dense_units, activation='relu')(x)
        outputs = Dense(CONFIG.galaxy10_decals_num_categories, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)

        if compile_model:
            model.compile(
                optimizer=Adam(learning_rate=self._dl_model.base_learning_rate),
                loss=CategoricalCrossentropy(),
                metrics=[CategoricalAccuracy()],
            )
        logger.info(model.summary())
        return model

    # TODO: Handle an already trained model
    def train(self):
        # Create the image generators and iterators
        aug_datagen, std_datagen = self.get_generators()
        dataset_manager = DatasetManager(self._dl_model.dataset)
        train_it, test_it = dataset_manager.get_iterators(aug_datagen, std_datagen, self._dl_model.batch_size)

        # Create the model
        model = self.get_model(compile_model=True)

        # Define the callbacks
        save_path = DLModelManager.get_save_path(self._dl_model)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        logs_path = DLModelManager.get_logs_path(self._dl_model)
        os.makedirs(os.path.dirname(logs_path), exist_ok=True)
        callbacks = [
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                min_delta=0.00005,
                patience=self._dl_model.reduce_learning_rate_patience, min_lr=1e-8, mode='min',
                verbose=2
            ),
            ModelCheckpoint(
                filepath=save_path,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                save_weights_only=True
            ),
            TensorBoard(
                log_dir=logs_path,
                histogram_freq=1
            )
        ]

        # Train the model
        model.fit(
            train_it,
            epochs=self._dl_model.epochs,
            validation_data=test_it,
            callbacks=callbacks
        )
        logger.info(f'DLModel {self._dl_model.id} trained')

    def evaluate(self, batch_size):
        # Create the image generators and iterators
        _, std_datagen = self.get_generators()
        dataset_manager = DatasetManager(self._dl_model.dataset)
        train_it, test_it = dataset_manager.get_iterators(std_datagen, std_datagen, batch_size, shuffle=False)

        # Create the model
        model = self.get_model(compile_model=True)

        # Load the best epoch
        save_path = DLModelManager.get_save_path(self._dl_model)
        model.load_weights(save_path)

        # Evaluate the model
        if self._dl_model.dataset.compressed:
            train_true = train_it.y
            test_true = test_it.y
        else:
            train_true = train_it.classes
            test_true = test_it.classes
        train_pred = model.predict(train_it)
        test_pred = model.predict(test_it)

        return train_true, train_pred, test_true, test_pred

    def predict(self, image: np.ndarray):
        # Create the model
        model = self.get_model(compile_model=False)

        # Load the best epoch
        save_path = DLModelManager.get_save_path(self._dl_model)
        model.load_weights(save_path)

        # Prepare the image
        image = Image.fromarray(image).resize(
            (self._dl_model.dataset.image_size, self._dl_model.dataset.image_size)
        )
        image = np.array(image)
        image = DLModelManager.scale_image(image)
        image = np.expand_dims(image, axis=0)

        # Predict the image
        probabilities_list = model.predict(image)
        probabilities = probabilities_list[0]
        category = np.argmax(probabilities, axis=0)

        return probabilities, category
