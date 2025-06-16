from PIL import Image
import os
import glob

def move_img_without_exif(oldpath, newpath):
    # From https://stackoverflow.com/questions/19786301/python-remove-exif-info-from-images
    image = Image.open(oldpath)

    data = list(image.getdata())
    image_without_exif = Image.new(image.mode, image.size)
    image_without_exif.putdata(data)
        
    image_without_exif.save(newpath)

    image_without_exif.close()


if __name__ == "__main__":
    for oldpath in glob.glob("asl_data_/*/*/*"):
        if "DS_Store" in oldpath:
            continue

        paths = oldpath.split("/")
        paths[0] = "asl_data"
        newpath = "/".join(paths[:-1])
        os.makedirs(newpath, exist_ok=True)
        newpath = "/".join(paths)
        move_img_without_exif(oldpath, newpath)

