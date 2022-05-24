import os
import matplotlib.pyplot as plt

from intelligent_placer import check_image

for dir_name in ['C:\\Users\\Admin\\Desktop\\PyProj\\Placer\\N_answer\\', 'C:\\Users\\Admin\\Desktop\\PyProj\\Placer\\Y_answer\\']:
    for img_file in os.listdir(dir_name):
        ans, res = check_image(dir_name + img_file)
        if ans:
            plt.imshow(res)
            plt.show()
        else:
            print(img_file + '  :(')
