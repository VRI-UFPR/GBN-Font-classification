from PIL import Image, ImageEnhance
import os
import random

path = "../datasets/base_datasets/dataset_aug"
fonts = ["Kanzlei", "Italic", "Script", "Antiqua"]

for font in fonts:
    image_path = os.path.join(path,font)
    
    for f in os.listdir(image_path):
        Original_Image = Image.open(os.path.join(image_path, f))
        
        for i in range(3):
            # rotated_image = Original_Image.rotate(random.randint(0, 30))
            
            # rotated_image.save(f'{os.path.join(image_path,f.split(".")[0])}_rotated_{i}.png')
            
            filter = ImageEnhance.Brightness(Original_Image)
            new_image = filter.enhance(random.uniform(0.5, 1.9))
            
            new_image.save(f'{os.path.join(image_path,f.split(".")[0])}_brightness_{i}.png')
            
            filter = ImageEnhance.Contrast(Original_Image)
            new_image =  filter.enhance(random.uniform(0.5, 1.9))
            
            new_image.save(f'{os.path.join(image_path,f.split(".")[0])}_contrast_{i}.png')
            
            filter = ImageEnhance.Sharpness(Original_Image)
            new_image =  filter.enhance(random.uniform(0.5, 1.9))
            
            new_image.save(f'{os.path.join(image_path,f.split(".")[0])}_sharpness_{i}.png')
            
            filter = ImageEnhance.Color(Original_Image)
            new_image =  filter.enhance(random.uniform(0.5, 1.9))
            
            new_image.save(f'{os.path.join(image_path,f.split(".")[0])}_color_{i}.png')
            
            
            
            
        