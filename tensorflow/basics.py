import tensorflow as tf

v = tf.Variable(
    initial_value=1.,
    trainable=True,
    name='variable',
    dtype=tf.float64,
    shape=None
)

print(v)

class HiddenLayer(Layer):
    def __init__(self, num_outputs):
        super(HiddenLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):

        w_value = tf.random.normal(shape=[int(input_shape[-1]),
                                         self.num_outputs])
        bias_value = tf.zeros([self.num_outputs])

        self.W = tf.Variable(initial_value=w_value)
        self.b = tf.Variable(initial_value=bias_value)

    def call(self, inputs):
        product = tf.matmul(inputs, self.W)
        return product +self.b

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.h1 = HiddenLayer(num_outputs=128)
        self.h2 = HiddenLayer(num_outputs=128)
        self.out = HiddenLayer(num_outputs=1)
    def call(self, x):
        x = self.h1(x)
        x = self.h2(x)
        x = self.out(x)
        return x


model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(100,)))
model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.Dense(1))
model.summary()

inp = tf.keras.Input(shape=(100))
x = tf.keras.layers.Dense(128, activation='relu')(inp)
x = tf.keras.layers.Dense(128, activation='relu')(x)
out = tf.keras.layers.Dense(128)(x)
model = keras.Model(inputs=inp, outputs=out, name="my_model")


model.compile(loss='mse', optimizer='adam', metrics='mse')

hist = model.fit(x, y, epochs=10000,
                 callbacks=[a_callback, another_super_callback],
                 validation_split=0.3)
