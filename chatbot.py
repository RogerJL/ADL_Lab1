import data_loading_code

vocab_size, (train_x_tensor, train_y_tensor), (validation_x_tensor, validation_y_tensor) = data_loading_code.load()
print(vocab_size)