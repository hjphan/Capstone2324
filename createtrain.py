import glob
images_list = glob.glob("data/obj/*.jpg")
print(images_list)
#Create training.txt file
file = open("data/train.txt", "w")
file.write("\n".join(images_list))
file.close()