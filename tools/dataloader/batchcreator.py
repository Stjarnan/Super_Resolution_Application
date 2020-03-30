import cv2
import numpy as np
import random

class BATCH:

    @staticmethod
    def get_batch(list_of_images, batch_size, high_resolution_shape, low_resolution_shape):

        # Choose a random batch of images
        images_batch = random.sample(list_of_images, batch_size)

        low_res = []
        high_res = []

        for img in images_batch:
            img = img.astype(np.float32)

            # Resize the image
            high_res_img = cv2.resize(img, high_resolution_shape)
            low_res_img = cv2.resize(img, low_resolution_shape)

            high_res.append(high_res_img)
            low_res.append(low_res_img)



        # Convert the lists to Numpy NDArrays
        return np.array(high_res), np.array(low_res)
