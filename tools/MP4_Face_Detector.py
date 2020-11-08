import cv2 as cv

face_cascade = cv.CascadeClassifier(
    r'C:\Users\micel\OneDrive\Documents\CodeSpace\Native\PyDrone\CVTools\haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)
cap.open(r'C:\Users\micel\OneDrive\Documents\CodeSpace\Native\PyDrone\CVTools\Video Of People Walking.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    img = frame
    scale_percent = 40  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    cv.imshow('frame', resized)
    if cv.waitKey(24) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
