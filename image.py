from PIL import Image as pil_image
import math
import numpy as np

from pygame import gfxdraw as g


class Image:
	def __init__(self, image_path):
		try:
			self.__img = pil_image.open(image_path).copy()
			# self.__img = self.__img
		except:
			self.__img = None

	@staticmethod
	def from_np_array(image_as_np_array, should_scale=False):
		if should_scale:
			image_as_np_array = image_as_np_array*255
		image_as_np_array = image_as_np_array
		img = Image(None)
		img.__img = pil_image.fromarray(image_as_np_array.astype(dtype=np.uint8)).copy()
		return img

	@staticmethod
	def from_list(image_as_list, dim_x, dim_y, is_rgb=False, should_scale=False):
		if is_rgb:
			image_as_np_array = np.reshape(image_as_list, (dim_y, dim_x, 3))
			if should_scale:
				image_as_np_array = image_as_np_array*255
				image_as_np_array = image_as_np_array
			img = Image(None)
			img.__img = pil_image.fromarray(image_as_np_array.astype(dtype=np.uint8), 'RGB').copy()
			return img
		else:
			image_as_np_array = np.reshape(image_as_list, (dim_y, dim_x))
			if should_scale:
				image_as_np_array = image_as_np_array*255
				image_as_np_array = image_as_np_array
			img = Image(None)
			img.__img = pil_image.fromarray(image_as_np_array.astype(dtype=np.uint8), 'L').copy()
			return img

	def show(self):
		self.__img.show()

	def convert_to_greyscale(self):
		self.__img = self.__img.convert('L')

	def get_greyscale(self):
		greyscale_image = Image(None)
		greyscale_image.__img = self.__img.convert('L')
		return greyscale_image

	def flip_left_right(self):
		self.__img = self.__img.transpose(pil_image.FLIP_LEFT_RIGHT)

	def flip_top_botton(self):
		self.__img = self.__img.transpose(pil_image.FLIP_TOP_BOTTOM)

	def resize(self, width, height, crop_loc='cc', resample=pil_image.BICUBIC):
		# crop_loc vals = cc, tr, tl, br, bl, tc, bc, cl, cr
		img_proportion = self.__img.height/self.__img.width
		new_img_proportion = height/width
		if new_img_proportion > img_proportion:
			# HEIGHT DOMINANT
			width_proportional = int(round(height*self.__img.width/self.__img.height))
			self.__img = self.__img.resize((width_proportional, height), resample)
			if crop_loc == 'cc':
				width_center = int(round(self.__img.width/2.0))
				delta_left = int(math.ceil(width/2.0))
				delta_right = int(math.floor(width/2.0))
				crop_area = (width_center - delta_left, 0, width_center + delta_right, self.__img.height)
				self.__img = self.__img.crop(crop_area)
		else:
			# WIDTH DOMINANT
			height_proportional = int(round(width*self.__img.height/self.__img.width))
			self.__img = self.__img.resize((width, height_proportional), resample)
			if crop_loc == 'cc':
				height_center = int(round(self.__img.height/2.0))
				delta_up = int(math.ceil(height/2.0))
				delta_down = int(math.floor(height/2.0))
				crop_area = (0, height_center - delta_up, self.__img.width, height_center + delta_down)
				self.__img = self.__img.crop(crop_area)

	def get_resized(self, width, height, crop_loc='cc', resample=pil_image.BICUBIC):
		# crop_loc vals = cc, tr, tl, br, bl, tc, bc, cl, cr
		resized_image = Image(None)
		img_proportion = self.__img.height/self.__img.width
		new_img_proportion = height/width
		if new_img_proportion > img_proportion:
			# HEIGHT DOMINANT
			width_proportional = int(round(height*self.__img.width/self.__img.height))
			self.__img = self.__img.resize((width_proportional, height), resample)
			if crop_loc == 'cc':
				width_center = int(round(self.__img.width/2.0))
				delta_left = int(math.ceil(width/2.0))
				delta_right = int(math.floor(width/2.0))
				crop_area = (width_center - delta_left, 0, width_center + delta_right, self.__img.height)
				resized_image.__img = self.__img.crop(crop_area)
				return resized_image
		else:
			# WIDTH DOMINANT
			height_proportional = int(round(width*self.__img.height/self.__img.width))
			self.__img = self.__img.resize((width, height_proportional), resample)
			if crop_loc == 'cc':
				height_center = int(round(self.__img.height/2.0))
				delta_up = int(math.ceil(height/2.0))
				delta_down = int(math.floor(height/2.0))
				crop_area = (0, height_center - delta_up, self.__img.width, height_center + delta_down)
				resized_image.__img = self.__img.crop(crop_area)
				return resized_image

	def save(self, filepath):
		self.__img.save(filepath)

	def get_mode(self):
		return self.__img.mode

	def get_as_numpy_array(self):
		return np.array(self.__img)

	def get_as_list(self):
		return np.array(self.__img).flatten().tolist()

	def get_as_list_scaled(self):
		return (np.array(self.__img)/255.0).flatten().tolist()

	def get_width(self):
		return self.__img.width

	def get_height(self):
		return self.__img.height

	def draw(self, w, x, y, width=None, height=None):
		# g.rectangle(w, (x, y, width, height), (255, 0, 0))
		if width is None and height is None:
			# DRAW TO SCALE
			width = self.__img.width
			height = self.__img.height
		elif width is None:
			# DRAW HEIGHT FIXED
			width = int(round(height*self.__img.width/self.__img.height))
			g.rectangle(w, (x, y, width, height), (255, 0, 0))
		elif height is None:
			# DRAW WIDTH FIXED
			height = int(round(width*self.__img.height/self.__img.width))
			g.rectangle(w, (x, y, width, height), (255, 0, 0))
		pixel_width = width/self.__img.width
		pixel_height = height/self.__img.height
		for i in range(self.__img.height):
			for j in range(self.__img.width):
				p_x = int(round(x + j*pixel_width))
				p_y = int(round(y + i*pixel_width))
				p_width = int(math.ceil(pixel_width))
				p_height = int(math.ceil(pixel_height))
				if self.__img.mode == 'RGB':
					p_color = self.__img.getpixel((j, i))
				else:
					value = self.__img.getpixel((j, i))
					p_color = (value, value, value)
				g.box(w, (p_x, p_y, p_width, p_height), p_color)
