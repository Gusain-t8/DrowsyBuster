from flask import Flask, render_template, Response, request
from pygame import mixer
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from twilio.rest import Client

# Initialize the mixer and load the alarm sound
mixer.init()
mixer.music.load("assets/Beeping-noise.wav")


# Define the eye aspect ratio function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


# Define the final eye aspect ratio function
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)


# Define the lip distance function
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


# Create a Flask web application
app = Flask(__name__)

# Constants for drowsiness detection
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
COUNTER = 0


# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())


# Define the drowsiness detection function
def drowsiness_detection():
    global alarm_status, alarm_status2, COUNTER, vs

    # Load the face detector and shape predictor
    print("-> Loading the predictor and detector...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    # Start the video stream
    print("-> Starting Video Stream")
    vs = VideoStream(
        src=args["webcam"], usePiCamera=False, backend=cv2.CAP_DSHOW
    ).start()
    time.sleep(1.0)

    # Twilio API credentials
    account_sid = "ACc51ce80eee4fedeab1a81b53b0f84df6"
    auth_token = "757acda04c6a31e5b7f26ce017e5c105"
    client = Client(account_sid, auth_token)
    phone_number = "XXX"  # Enter the recipient's phone number

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            eye = final_ear(shape)
            ear = eye[0]
            leftEye = eye[1]
            rightEye = eye[2]

            distance = lip_distance(shape)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not alarm_status:
                        alarm_status = True
                        mixer.music.play()
                        # Send SMS alert
                        message = client.messages.create(
                            body="Driver is drowsy! Please take necessary action.",
                            from_="xxx",
                            to=phone_number,
                        )

                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT!",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            else:
                COUNTER = 0
                alarm_status = False

            if distance > YAWN_THRESH:
                cv2.putText(
                    frame,
                    "Yawn Alert",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                if not alarm_status2:
                    alarm_status2 = True
                    mixer.music.play()
                    # Send SMS alert
                    message = client.messages.create(
                        body="Driver is yawning! Please take necessary action.",
                        from_="xxx",
                        to=phone_number,
                    )
            else:
                alarm_status2 = False

            cv2.putText(
                frame,
                "EAR: {:.2f}".format(ear),
                (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "YAWN: {:.2f}".format(distance),
                (300, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        # yield statement the frames are continuously streamed to the web browser
        ret, jpeg = cv2.imencode(".jpg", frame)
        frame_bytes = jpeg.tobytes()
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


# Define the index route
@app.route("/")
def index():
    return render_template("index.html")


# defining video feed route
@app.route("/video_feed")
def video_feed():
    return Response(
        drowsiness_detection(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# defining start route
@app.route("/start", methods=["POST"])
def start():
    drowsiness_detection()
    mixer.music.stop()
    return "Success"


# defining stop route
@app.route("/stop", methods=["POST"])
def stop():
    mixer.music.stop()
    return "Success"


# run the flask app
if __name__ == "__main__":
    app.run(debug=True)
