import cv2 as cv

face_cascade = cv.CascadeClassifier(
    r'C:\Users\micel\OneDrive\Documents\CodeSpace\Native\PyDrone\CVTools\haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(
    r'C:\Users\micel\OneDrive\Documents\CodeSpace\Native\PyDrone\CVTools\haarcascade_eye.xml')

cap = cv.VideoCapture(0)
print(cap.isOpened())

while(True):
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.15, 5)
    print(len(faces))

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        eye_gray = gray[y:y+h, x:x+w]
        eye_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(eye_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(eye_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
