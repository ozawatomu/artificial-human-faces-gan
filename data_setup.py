import os
from image import Image

i = 0
for dir_path, dir_names, file_names in os.walk('src/face_data/cfp-dataset/cfp-dataset/Data/Images'):
	if dir_path.endswith('frontal'):
		for file_name in file_names:
			img = Image('{}/{}'.format(dir_path, file_name))
			img.convert_to_greyscale()
			img.resize(64, 64)
			name = f'{i:04}'
			img.save('src/training_data/{}.jpg'.format(name))
			i += 1
			img.flip_left_right()
			name = f'{i:04}'
			img.save('src/training_data/{}.jpg'.format(name))
			i += 1