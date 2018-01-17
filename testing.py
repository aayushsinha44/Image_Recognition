def display_class(path):
    # load json and create model
    from keras.models import model_from_json
    from keras.optimizers import SGD
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    print (path)
    epochs = 25
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img(path, target_size = (32, 32))
    test_image = image.img_to_array(test_image)
    test_image = np.transpose(test_image, (2,0,1))
    test_image = np.expand_dims(test_image, axis = 0)
    result = loaded_model.predict(test_image)

    # search for the class
    for i in range(0,10):
        if result[0][i] == 1.0:
            index = i
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return (classes[index])

if __name__ == '__main__':
    image_type = display_class("C:\\Users\\Aayush\\Desktop\\Object_Recognition\\dog.jpg")
    print (image_type)