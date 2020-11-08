import cv2 as cv

face_cascade = cv.CascadeClassifier(
    r'tools\haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(
    r'tools\haarcascade_eye.xml')

img = cv.imread(
    r'tools\image1.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.15, 5)
print(len(faces))

for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    eye_gray = gray[y:y+h, x:x+w]
    eye_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(eye_gray)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(eye_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

scale_percent = 40  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

print('Resized Dimensions : ', resized.shape)

cv.imshow("Resized image", resized)
cv.waitKey(0)
cap.release()
cv.destroyAllWindows()
