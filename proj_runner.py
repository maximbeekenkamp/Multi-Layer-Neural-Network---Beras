from types import SimpleNamespace

import numpy as np

import Beras


class SequentialModel(Beras.Model):
    """
    Implemented in Beras/model.py

    def __init__(self, layers):
    def compile(self, optimizer, loss_fn, acc_fn):
    def fit(self, x, y, epochs, batch_size):
    def evaluate(self, x, y, batch_size):
    """

    def call(self, inputs):
        """
        Forward pass in sequential model. It's helpful to note that layers are initialized in Beras.Model, and
        you can refer to them with self.layers. You can call a layer by doing var = layer(input).
        """
        x = np.copy(inputs)
        for layer in self.layers:
            x = layer(x)
        return x

    def batch_step(self, x, y, training=True):
        """
        Computes loss and accuracy for a batch. This step consists of both a forward and backward pass.
        If training=false, don't apply gradients to update the model!
        Most of this method (forward, loss, applying gradients)
        will take place within the scope of Beras.GradientTape()
        """
        with Beras.GradientTape() as tape:
            logits = self.call(x)
            loss = self.compiled_loss(logits, y)
            if training:
                grads = tape.gradient()
                self.optimizer.apply_gradients(self.trainable_variables, grads)
        acc = self.compiled_acc(logits, y)

        return {"loss": loss, "acc": acc}


def get_simple_model_components():
    """
    Returns a simple single-layer model.
    """

    from Beras.activations import Softmax, LeakyReLU, Sigmoid
    from Beras.layers import Dense
    from Beras.losses import MeanSquaredError
    from Beras.metrics import CategoricalAccuracy
    from Beras.optimizers import BasicOptimizer, RMSProp, Adam

    model = SequentialModel([Dense(784, 10, "kaiming"), Softmax()])
    model.compile(
        optimizer=BasicOptimizer(0.05),
        loss_fn=MeanSquaredError(),
        acc_fn=CategoricalAccuracy(),
    )
    return SimpleNamespace(model=None, epochs=None, batch_size=None)


def get_advanced_model_components():
    """
    Returns a multi-layered model with more involved components.
    """
    
    from Beras.activations import Softmax, LeakyReLU
    from Beras.layers import Dense
    from Beras.losses import MeanSquaredError
    from Beras.metrics import CategoricalAccuracy
    from Beras.optimizers import BasicOptimizer, RMSProp, Adam

    model = SequentialModel([Dense(784, 150, "kaiming"), LeakyReLU(), Dense(150, 50, "kaiming"), LeakyReLU(), Dense(50, 10, "xavier"), Softmax()])
    model.compile(
        optimizer=BasicOptimizer(0.4),
        loss_fn=MeanSquaredError(),
        acc_fn=CategoricalAccuracy(),
    )

    return SimpleNamespace(model=None, epochs=None, batch_size=None)


if __name__ == "__main__":
    """
    Read in MNIST data and initialize/train/test your model.
    """
    import preprocess
    from Beras.onehot import OneHotEncoder

    ## Read in MNIST data,
    train_inputs, train_labels = preprocess.get_data_MNIST("train", "../data")
    test_inputs, test_labels = preprocess.get_data_MNIST("test", "../data")

    ohe = OneHotEncoder()
    ohe.fit(data=np.concatenate((train_labels, test_labels), axis=-1)) ## placeholder function: returns zero for a given input

    ## Get your model to train and test
    simple = True
    args = get_simple_model_components() if simple else get_advanced_model_components()
    model = args.model

    # Fits your model to the training input and the one hot encoded labels
    train_agg_metrics = model.fit(
        train_inputs, ohe(train_labels), epochs=args.epochs, batch_size=args.batch_size
    )

    test_agg_metrics = model.evaluate(test_inputs, ohe(test_labels), batch_size=100)
    print("Testing Performance:", test_agg_metrics)
