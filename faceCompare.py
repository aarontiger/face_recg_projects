import face_recognition
from PIL import Image
import matplotlib.pyplot as plt

#人脸比对
def faceCompare():
    known_image = face_recognition.load_image_file("images/haoqy1.jpg")
    unknown_image = face_recognition.load_image_file("images/haoqy2.jpg")
    img1 = Image.open("images/haoqy1.jpg")
    plt.imshow(img1)
    plt.axis('off')
    plt.show()
    img2 = Image.open("images/haoqy2.jpg")
    plt.imshow(img2)
    plt.axis('off')
    plt.show()
    baixiaona_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
    results = face_recognition.compare_faces([baixiaona_encoding], unknown_encoding)
    if results[0] == True:
        print("Yes!")
    else:
        print("No!")


if __name__ == "__main__":
    print("222")
    faceCompare()