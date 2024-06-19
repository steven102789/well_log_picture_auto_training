from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Conv1DTranspose, Dropout, concatenate

class CNNModel:
    def __init__(self, input_shape, n_classes, kernel_size):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.ks = kernel_size
        self.model = self.build_model()

    def build_model(self):
        # Define input layers
        n16_inputs = Input(shape=(self.input_shape, 1), name="Input_n16")
        n64_inputs = Input(shape=(self.input_shape, 1), name="Input_n64")
        gamma_inputs = Input(shape=(self.input_shape, 1), name="Input_gamma")
        sp_inputs = Input(shape=(self.input_shape, 1), name="Input_sp")

        # Build Contraction path
        # n16
        n16_c1 = Conv1D(filters=16, kernel_size=self.ks, activation='relu', padding='same')(n16_inputs)
        n16_c1 = Dropout(0.1)(n16_c1)
        n16_c1 = Conv1D(filters=16, kernel_size=self.ks, activation='relu', padding='same')(n16_c1)
        n16_p1 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(n16_c1)

        n16_c2 = Conv1D(filters=32, kernel_size=self.ks, activation='relu', padding='same')(n16_p1)
        n16_c2 = Dropout(0.1)(n16_c2)
        n16_c2 = Conv1D(filters=32, kernel_size=self.ks, activation='relu', padding='same')(n16_c2)
        n16_p2 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(n16_c2)

        n16_c3 = Conv1D(filters=64, kernel_size=self.ks, activation='relu', padding='same')(n16_p2)
        n16_c3 = Dropout(0.2)(n16_c3)
        n16_c3 = Conv1D(filters=64, kernel_size=self.ks, activation='relu', padding='same')(n16_c3)
        n16_p3 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(n16_c3)

        n16_c4 = Conv1D(filters=128, kernel_size=self.ks, activation='relu', padding='same')(n16_p3)
        n16_c4 = Dropout(0.2)(n16_c4)
        n16_c4 = Conv1D(filters=128, kernel_size=self.ks, activation='relu', padding='same')(n16_c4)
        n16_p4 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(n16_c4)

        n16_c5 = Conv1D(filters=256, kernel_size=self.ks, activation='relu', padding='same')(n16_p4)
        n16_c5 = Dropout(0.3)(n16_c5)
        n16_c5 = Conv1D(filters=256, kernel_size=self.ks, activation='relu', padding='same')(n16_c5)

        # n64
        n64_c1 = Conv1D(filters=16, kernel_size=self.ks, activation='relu', padding='same')(n64_inputs)
        n64_c1 = Dropout(0.1)(n64_c1)
        n64_c1 = Conv1D(filters=16, kernel_size=self.ks, activation='relu', padding='same')(n64_c1)
        n64_p1 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(n64_c1)

        n64_c2 = Conv1D(filters=32, kernel_size=self.ks, activation='relu', padding='same')(n64_p1)
        n64_c2 = Dropout(0.1)(n64_c2)
        n64_c2 = Conv1D(filters=32, kernel_size=self.ks, activation='relu', padding='same')(n64_c2)
        n64_p2 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(n64_c2)

        n64_c3 = Conv1D(filters=64, kernel_size=self.ks, activation='relu', padding='same')(n64_p2)
        n64_c3 = Dropout(0.2)(n64_c3)
        n64_c3 = Conv1D(filters=64, kernel_size=self.ks, activation='relu', padding='same')(n64_c3)
        n64_p3 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(n64_c3)

        n64_c4 = Conv1D(filters=128, kernel_size=self.ks, activation='relu', padding='same')(n64_p3)
        n64_c4 = Dropout(0.2)(n64_c4)
        n64_c4 = Conv1D(filters=128, kernel_size=self.ks, activation='relu', padding='same')(n64_c4)
        n64_p4 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(n64_c4)

        n64_c5 = Conv1D(filters=256, kernel_size=self.ks, activation='relu', padding='same')(n64_p4)
        n64_c5 = Dropout(0.3)(n64_c5)
        n64_c5 = Conv1D(filters=256, kernel_size=self.ks, activation='relu', padding='same')(n64_c5)

        # gamma
        gamma_c1 = Conv1D(filters=16, kernel_size=self.ks, activation='relu', padding='same')(gamma_inputs)
        gamma_c1 = Dropout(0.1)(gamma_c1)
        gamma_c1 = Conv1D(filters=16, kernel_size=self.ks, activation='relu', padding='same')(gamma_c1)
        gamma_p1 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(gamma_c1)

        gamma_c2 = Conv1D(filters=32, kernel_size=self.ks, activation='relu', padding='same')(gamma_p1)
        gamma_c2 = Dropout(0.1)(gamma_c2)
        gamma_c2 = Conv1D(filters=32, kernel_size=self.ks, activation='relu', padding='same')(gamma_c2)
        gamma_p2 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(gamma_c2)

        gamma_c3 = Conv1D(filters=64, kernel_size=self.ks, activation='relu', padding='same')(gamma_p2)
        gamma_c3 = Dropout(0.2)(gamma_c3)
        gamma_c3 = Conv1D(filters=64, kernel_size=self.ks, activation='relu', padding='same')(gamma_c3)
        gamma_p3 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(gamma_c3)

        gamma_c4 = Conv1D(filters=128, kernel_size=self.ks, activation='relu', padding='same')(gamma_p3)
        gamma_c4 = Dropout(0.2)(gamma_c4)
        gamma_c4 = Conv1D(filters=128, kernel_size=self.ks, activation='relu', padding='same')(gamma_c4)
        gamma_p4 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(gamma_c4)

        gamma_c5 = Conv1D(filters=256, kernel_size=self.ks, activation='relu', padding='same')(gamma_p4)
        gamma_c5 = Dropout(0.3)(gamma_c5)
        gamma_c5 = Conv1D(filters=256, kernel_size=self.ks, activation='relu', padding='same')(gamma_c5)

        # sp
        sp_c1 = Conv1D(filters=16, kernel_size=self.ks, activation='relu', padding='same')(sp_inputs)
        sp_c1 = Dropout(0.1)(sp_c1)
        sp_c1 = Conv1D(filters=16, kernel_size=self.ks, activation='relu', padding='same')(sp_c1)
        sp_p1 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(sp_c1)

        sp_c2 = Conv1D(filters=32, kernel_size=self.ks, activation='relu', padding='same')(sp_p1)
        sp_c2 = Dropout(0.1)(sp_c2)
        sp_c2 = Conv1D(filters=32, kernel_size=self.ks, activation='relu', padding='same')(sp_c2)
        sp_p2 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(sp_c2)

        sp_c3 = Conv1D(filters=64, kernel_size=self.ks, activation='relu', padding='same')(sp_p2)
        sp_c3 = Dropout(0.2)(sp_c3)
        sp_c3 = Conv1D(filters=64, kernel_size=self.ks, activation='relu', padding='same')(sp_c3)
        sp_p3 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(sp_c3)

        sp_c4 = Conv1D(filters=128, kernel_size=self.ks, activation='relu', padding='same')(sp_p3)
        sp_c4 = Dropout(0.2)(sp_c4)
        sp_c4 = Conv1D(filters=128, kernel_size=self.ks, activation='relu', padding='same')(sp_c4)
        sp_p4 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(sp_c4)

        sp_c5 = Conv1D(filters=256, kernel_size=self.ks, activation='relu', padding='same')(sp_p4)
        sp_c5 = Dropout(0.3)(sp_c5)
        sp_c5 = Conv1D(filters=256, kernel_size=self.ks, activation='relu', padding='same')(sp_c5)

        # Build Expansive path
        u6 = concatenate([n16_c5, n64_c5, gamma_c5, sp_c5])  # c5 is the code
        u6 = Conv1DTranspose(filters=128, kernel_size=2, activation='relu')(u6)
        # up sampling to the size which is same as block C4
        u6 = concatenate([u6, concatenate([n16_c4, n64_c4, gamma_c4, sp_c4])])
        c6 = Conv1D(filters=128, kernel_size=3, activation='relu')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv1D(filters=128, kernel_size=3, activation='relu')(c6)
        u7 = Conv1DTranspose(filters=64, kernel_size=6, activation='relu')(c6)
        u7 = concatenate([u7, concatenate([n16_c3, n64_c3, gamma_c3, sp_c3])])
        c7 = Conv1D(filters=64, kernel_size=3, activation='relu')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv1D(filters=64, kernel_size=3, activation='relu')(c7)

        u8 = Conv1DTranspose(filters=32, kernel_size=6, activation='relu')(c7)
        u8 = concatenate([u8, concatenate([n16_c2, n64_c2, gamma_c2, sp_c2])])
        c8 = Conv1D(filters=32, kernel_size=3, activation='relu')(u8)
        c8 = Dropout(0.2)(c8)
        c8 = Conv1D(filters=32, kernel_size=3, activation='relu')(c8)

        u9 = Conv1DTranspose(filters=16, kernel_size=6, activation='relu')(c8)
        u9 = concatenate([u9, concatenate([n16_c1, n64_c1, gamma_c1, sp_c1])])
        c9 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(u9)
        c9 = Dropout(0.2)(c9)
        c9 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(c9)

        # Define outputs
        outputs = Conv1D(self.n_classes, kernel_size=1, activation='softmax')(c9)

        # Create the merged model
        model = Model(inputs=[n16_inputs, n64_inputs, gamma_inputs, sp_inputs], outputs=[outputs], name="MergedModel")
        return model

    def summary(self):
        self.model.summary()

__all__ = ["CNNModel"]