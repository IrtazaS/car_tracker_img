import cv2

# load image with cars or pedestrians
img_file = "images/cars.jpg"
#img_file = "images/pedestrians.jpg"

# load pre-trained car & pedestrians classifier
car_classifier_file = 'classifiers/car_classifier.xml'
pedestrian_classifier_file = 'classifiers/haarcascade_fullbody.xml'

#create opencv image
img = cv2.imread(img_file)

#convert to grayscale (needed for haar cascade)
#also faster to process
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create car and pedetrian classifier
car_tracker = cv2.CascadeClassifier(car_classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier_file)

#detect cars and pedestrians
cars = car_tracker.detectMultiScale(black_n_white)
pedestrians = pedestrian_tracker.detectMultiScale(black_n_white)

print("Detected cars:")
print(cars)
print("Detected pedestrians:")
print(pedestrians)

# draw rects around detected cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # draw rects around detected cars
for (x, y, w, h) in pedestrians:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#Display image
cv2.imshow("Cars and Pedestrians detected by Classifier", img)

#Wait and listen for key press (prevents autoclose)
cv2.waitKey()

print("No errors!")
