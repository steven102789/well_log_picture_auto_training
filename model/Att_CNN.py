from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Dropout, concatenate, Conv1D, Conv1DTranspose, MaxPooling1D
class CrossAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(CrossAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.add = layers.Add()

    def call(self, inputs):
        query, key, value = inputs
        attn_output = self.attention(query=query, key=key, value=value)
        output = self.add([query, attn_output])
        return self.norm(output)


    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads
        })
        return config

class AttCNN(layers.Layer):
    def __init__(self, input_shape, n_classes, kernel_size, embed_dim, num_heads):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.model = self.build_model()

    def build_model(self):
        # Define input layers
        n16_inputs = Input(shape=(self.input_shape, 1), name="Input_n16")
        n64_inputs = Input(shape=(self.input_shape, 1), name="Input_n64")
        gamma_inputs = Input(shape=(self.input_shape, 1), name="Input_gamma")
        sp_inputs = Input(shape=(self.input_shape, 1), name="Input_sp")

        # Define shared layers
        def conv_block(inputs, filters, kernel_size, dropout_rate):
            conv = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(inputs)
            conv = Dropout(dropout_rate)(conv)
            conv = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(conv)
            return conv

        def contraction_path(inputs):
            c1 = conv_block(inputs, 16, self.kernel_size, 0.1)
            p1 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(c1)
            c2 = conv_block(p1, 32, self.kernel_size, 0.1)
            p2 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(c2)
            c3 = conv_block(p2, 64, self.kernel_size, 0.2)
            p3 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(c3)
            c4 = conv_block(p3, 128, self.kernel_size, 0.2)
            p4 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(c4)
            c5 = conv_block(p4, 256, self.kernel_size, 0.3)
            return c1, c2, c3, c4, c5
        # Create contraction paths for each input
        n16_c1, n16_c2, n16_c3, n16_c4, n16_c5 = contraction_path(n16_inputs)
        n64_c1, n64_c2, n64_c3, n64_c4, n64_c5 = contraction_path(n64_inputs)
        gamma_c1, gamma_c2, gamma_c3, gamma_c4, gamma_c5 = contraction_path(gamma_inputs)
        sp_c1, sp_c2, sp_c3, sp_c4, sp_c5 = contraction_path(sp_inputs)

        # Implementing Cross Attention
        cross_attention = CrossAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)

        # Cross Attention for each pair of modalities
        n16_cross_n64 = cross_attention(n16_c5, n64_c5, n64_c5)
        n16_cross_gamma = cross_attention(n16_c5, gamma_c5, gamma_c5)
        n16_cross_sp = cross_attention(n16_c5, sp_c5, sp_c5)

        n64_cross_n16 = cross_attention(n64_c5, n16_c5, n16_c5)
        n64_cross_gamma = cross_attention(n64_c5, gamma_c5, gamma_c5)
        n64_cross_sp = cross_attention(n64_c5, sp_c5, sp_c5)

        gamma_cross_n16 = cross_attention(gamma_c5, n16_c5, n16_c5)
        gamma_cross_n64 = cross_attention(gamma_c5, n64_c5, n64_c5)
        gamma_cross_sp = cross_attention(gamma_c5, sp_c5, sp_c5)

        sp_cross_n16 = cross_attention(sp_c5, n16_c5, n16_c5)
        sp_cross_n64 = cross_attention(sp_c5, n64_c5, n64_c5)
        sp_cross_gamma = cross_attention(sp_c5, gamma_c5, gamma_c5)

        # Concatenate cross attention outputs
        cross_outputs = concatenate([
            n16_cross_n64, n16_cross_gamma, n16_cross_sp,
            n64_cross_n16, n64_cross_gamma, n64_cross_sp,
            gamma_cross_n16, gamma_cross_n64, gamma_cross_sp,
            sp_cross_n16, sp_cross_n64, sp_cross_gamma
        ])

        # Build Expansive path
        u6 = Conv1DTranspose(filters=128, kernel_size=2, strides=1, activation='relu')(cross_outputs)
        u6 = concatenate([u6, n16_c4, n64_c4, gamma_c4, sp_c4])
        c6 = Conv1D(filters=128, kernel_size=3, activation='relu')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv1D(filters=128, kernel_size=3, activation='relu')(c6)

        u7 = Conv1DTranspose(filters=64, kernel_size=6, strides=1, activation='relu')(c6)
        u7 = concatenate([u7, n16_c3, n64_c3, gamma_c3, sp_c3])
        c7 = Conv1D(filters=64, kernel_size=3, activation='relu')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv1D(filters=64, kernel_size=3, activation='relu')(c7)

        u8 = Conv1DTranspose(filters=32, kernel_size=6, strides=1, activation='relu')(c7)
        u8 = concatenate([u8, n16_c2, n64_c2, gamma_c2, sp_c2])
        c8 = Conv1D(filters=32, kernel_size=3, activation='relu')(u8)
        c8 = Dropout(0.2)(c8)
        c8 = Conv1D(filters=32, kernel_size=3, activation='relu')(c8)

        u9 = Conv1DTranspose(filters=16, kernel_size=6, strides=1, activation='relu')(c8)
        u9 = concatenate([u9, n16_c1, n64_c1, gamma_c1, sp_c1])
        c9 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(u9)
        c9 = Dropout(0.2)(c9)
        c9 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(c9)

        # Define outputs
        outputs = Conv1D(self.n_classes, kernel_size=1, activation='softmax')(c9)

        # Create the merged model
        model = Model(inputs=[n16_inputs, n64_inputs, gamma_inputs, sp_inputs], outputs=[outputs],
                      name="MergedModel")
        return model
    def summary(self):
        self.model.summary()

__all__ = ["CNNModel.py"]