import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


path = 'F:\dpL\Defect_Least\Training\Roots'
target_path = 'F:\dpL\Defect_Least\Training\Roots'

datagen = ImageDataGenerator(
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')

data_dir = os.listdir(path)

ite = 0
count = 0


while count <= 1448:
    gap = 3
    print(ite)
    image = data_dir[count]
    print(image)
    img = load_img(path + '//' + image)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=target_path, save_prefix='Roots_',
                              save_format='jpeg'):
        i += 1
        if i > 1:
            break
    count += 4
    ite += 1
