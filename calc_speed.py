from datetime import datetime
import cv2
import math
import time
from picamera import PiCamera
from exif import Image
import statistics as s

GSD = 12648
FEATURE_NUMBER = 750
IMAGE_PREFIX = 'photo'
NUM_IMAGES = 25

def get_time(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        times = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return times

def get_time_difference(image_1, image_2):
    return (get_time(image_2) - get_time(image_1)).seconds

def convert_to_cv(image):
    return cv2.imread(image, cv2.IMREAD_GRAYSCALE)

def calculate_features(image_cv):
    orb = cv2.ORB_create(nfeatures=FEATURE_NUMBER)
    return orb.detectAndCompute(image_cv, None)

def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    return sorted(matches, key=lambda x: x.distance)

def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1, y1) = keypoints_1[image_1_idx].pt
        (x2, y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1, y1))
        coordinates_2.append((x2, y2))
    return coordinates_1, coordinates_2

def calculate_mean_distance(coordinates_1, coordinates_2):
    distances = [math.hypot(coord1[0] - coord2[0], coord1[1] - coord2[1])
                 for coord1, coord2 in zip(coordinates_1, coordinates_2)]
    return s.mean(distances)

def calculate_speed_in_kmps(image_1, image_2):
    time_difference = get_time_difference(image_1, image_2)
    image_1_cv, image_2_cv = convert_to_cv(image_1), convert_to_cv(image_2)
    keypoints_1, descriptors_1 = calculate_features(image_1_cv)
    keypoints_2, descriptors_2 = calculate_features(image_2_cv)
    matches = calculate_matches(descriptors_1, descriptors_2)
    coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
    average_distance = calculate_mean_distance(coordinates_1, coordinates_2)
    distance = average_distance * GSD / 100000
    return distance / time_difference

def capture_image(file_path):
    with PiCamera() as camera:
        camera.resolution = (1280,720)
        camera.capture(file_path)
        time.sleep(1)

def calculate_and_average_speed():
    speeds = []
    for i in range(1, NUM_IMAGES):
        image_1 = f"{IMAGE_PREFIX}{i}.jpg"
        image_2 = f"{IMAGE_PREFIX}{i + 1}.jpg"
        speed = calculate_speed_in_kmps(image_1, image_2)
        speeds.append(speed)
    return s.mean(speeds)

def main():
    start_time = time.time()
    end_time = start_time + 29 * 60  # Run for 29 minutes
    test_results = []

    while time.time() < end_time:
        for i in range(1, NUM_IMAGES + 1):
            image_path = f"{IMAGE_PREFIX}{i}.jpg"
            capture_image(image_path)
        average_speed = calculate_and_average_speed()
        test_results.append(average_speed)

    overall_average_speed = sum(test_results) / len(test_results)

    with open("result.txt", "w") as result_file:
        result_file.write(f"{overall_average_speed:.5f}")

if __name__ == "__main__":
    main()
