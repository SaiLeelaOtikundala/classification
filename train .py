x_train, x_test, y_train, y_test = train_test_split(images, label_data, test_size=0.2, random_state=42)

x_train = np.transpose(x_train, (0, 2, 3, 1))  # Convert to (batch_size, 32, 32, 3)
x_test = np.transpose(x_test, (0, 2, 3, 1))  # Convert to (batch_size, 32, 32, 3)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)