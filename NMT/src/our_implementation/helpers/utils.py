import keras.backend as K
from keras.engine.topology import Layer
from keras.layers.merge import multiply
from sklearn.metrics import log_loss


# Custom loss
class CustomLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomLossLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        self.add_loss(inputs, inputs=inputs)
        return inputs


def neg_log_likelihood(y_true, y_pred):
    probs = multiply([y_true, y_pred])
    probs = K.sum(probs, axis=-1)
    return 1e-06 + K.sum(-K.log(probs))


def categorical_cross_entropy(y_true, y_pred):
    print(y_true.shape, y_pred.shape)
    # y_true = K.reshape(y_true, (y_true.shape[0] * y_true.shape[1], y_true.shape[2]))
    # y_pred = K.reshape(y_pred, (y_pred.shape[0] * y_pred.shape[1], y_pred.shape[2]))
    return log_loss(y_true, y_pred)


# monitoring
def identity(y_true, y_pred):
    return y_pred


def zero(y_true, y_pred):
    return K.zeros((1,))


def print_dict_utf8(d):
    # iterate over the key/values pairings
    for k, v in d.items():
        # if v is a list join and encode else just encode as it is a string
        if isinstance(v, list):
            d[k] = ",".join(v).encode("utf-8")
        else:
            try:
                d[k] = v.encode("utf-8")
            except AttributeError:
                d[k] = v
    print(d)
