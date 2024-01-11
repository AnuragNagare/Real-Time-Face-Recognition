import argparse
import pickle
from collections import Counter
from pathlib import Path

import cv2
import face_recognition

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

parser = argparse.ArgumentParser(description="Recognize faces from the camera")
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for recognition: hog (CPU), cnn (GPU)",
)
args = parser.parse_args()


def _recognize_face(unknown_encoding, loaded_encodings):
    """
    Given an unknown encoding and all known encodings, find the known
    encoding with the most matches.
    """
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]


def access_camera_and_recognize(model: str = "hog"):
    with DEFAULT_ENCODINGS_PATH.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    # Access the camera
    video_capture = cv2.VideoCapture(0)  # 0 for default camera, change if needed

    while True:
        ret, frame = video_capture.read()

        # Convert frame from BGR to RGB for face_recognition
        rgb_frame = frame[:, :, ::-1]

        # Find face locations and encodings in the frame
        face_locations = face_recognition.face_locations(rgb_frame, model=model)
        face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations
        )

        for face_location, face_encoding in zip(
            face_locations, face_encodings
        ):
            name = _recognize_face(face_encoding, loaded_encodings)
            if not name:
                name = "Unknown"

            # Draw rectangle around the face and display the name
            top, right, bottom, left = face_location
            cv2.rectangle(
                frame, (left, top), (right, bottom), (0, 0, 255), 2
            )
            cv2.putText(
                frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1, cv2.LINE_AA
            )

        # Display the resulting frame
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    access_camera_and_recognize(model=args.m)
