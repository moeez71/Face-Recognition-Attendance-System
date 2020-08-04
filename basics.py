import cv2
import face_recognition

imgPerson = face_recognition.load_image_file('Students/A1.jpg')
imgPerson = cv2.cvtColor(imgPerson, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Students/A2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgPerson)[0]
encodeElon = face_recognition.face_encodings(imgPerson)[0]
cv2.rectangle(imgPerson, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

img1 = cv2.resize(imgPerson,(500,500))
img2 = cv2.resize(imgTest,(500,500))
cv2.imshow('Afridi ', img1)
cv2.imshow('Afridi Test Image', img2)
cv2.waitKey(0)