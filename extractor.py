from keras.preprocessing.image import img_to_array, load_img
from keras.applications.inception_v3 import InceptionV3, preprocess_input
# from keras.applications import MobileNetV2
# from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np

class Extractor():
    def __init__(self, image_shape=(299, 299, 3), weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model
        self.input_shape = image_shape
        input_tensor = Input(image_shape)

        # Get model with pretrained weights. #include_top -> fully connected layer at the end. 
        base_model = InceptionV3(
            input_tensor=input_tensor,
            weights='imagenet',
            include_top=True
        )
        # We'll extract features at the final pool layer.
        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('avg_pool').output
        )

        # base_model = MobileNetV2(
        #     input_tensor=input_tensor,
        #     weights='imagenet',
        #     include_top=True
        # )
        # self.model = Model(
        #     inputs=base_model.input,
        #     outputs=base_model.layers[-2].output
        # )

    def extract(self, image_path):
        img = load_img(image_path, target_size=self.input_shape)
        return self.extract_image(img)

    def extract_image(self, img):
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features