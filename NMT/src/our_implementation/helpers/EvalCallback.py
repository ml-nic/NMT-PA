import keras


class EvalCallback(keras.callbacks.Callback):
    def __init__(self, val_data_generator, val_steps: int, model_identifier: str, frequency: int = 1):
        super(EvalCallback, self).__init__()

        self.val_data_generator = val_data_generator
        self.validation_steps = val_steps
        self.model_identifier = model_identifier
        self.frequency = frequency
        self.current_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        self.current_epoch += 1
        if self.current_epoch % self.frequency == 0:
            print("now eval callback:")
            losses = 0.
            num = 0
            for i in range(self.validation_steps):
                x, y = next(self.val_data_generator)
                losses += self.model.evaluate(x, y, x.shape[0])
                num += 1
            losses = losses / num

            print("epoch:", epoch, "validation Loss:", losses)

            with(open('../../Persistence/' + self.model_identifier + '/val_data.txt', 'a')) as file:
                file.write("epoch" + str(epoch) + "realepoch" + str(epoch) + "val_loss" + str(losses))
