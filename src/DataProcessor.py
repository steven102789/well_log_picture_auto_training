import tensorflow as tf
class DataProcessor:
    def __init__(self, split_data, input_shape):
        self.input_shape = input_shape
        self.split_data = split_data

    def preprocess_data(self, key):
        data_list:list = list(self.split_data[key])
        return tf.reshape(data_list, (len(data_list), self.input_shape, 1))

