from exif import Image
from datetime import datetime
import cv2
import math
import time
from picamera import PiCamera
GSD = 12648


def get_time(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        times = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return times


def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1
    return time_difference.seconds


def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, cv2.IMREAD_GRAYSCALE)
    image_2_cv = cv2.imread(image_2, cv2.IMREAD_GRAYSCALE)
    return image_1_cv, image_2_cv


def calculate_features_parallel(image_cv, feature_number):
    orb = cv2.ORB_create(nfeatures=feature_number)
    key_points, descriptors = orb.detectAndCompute(image_cv, None)
    return key_points, descriptors


def calculate_features(image_1_cv, image_2_cv, feature_number):
    key_points_1, descriptors_1 = calculate_features_parallel(image_1_cv, feature_number)
    key_points_2, descriptors_2 = calculate_features_parallel(image_2_cv, feature_number)
    return key_points_1, key_points_2, descriptors_1, descriptors_2


def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def find_matching_coordinates(key_points_1, key_points_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1, y1) = key_points_1[image_1_idx].pt
        (x2, y2) = key_points_2[image_2_idx].pt
        coordinates_1.append((x1, y1))
        coordinates_2.append((x2, y2))
    return coordinates_1, coordinates_2


def calculate_mean_distance(coordinates_1, coordinates_2):
    all_distances = 0
    for coord1, coord2 in zip(coordinates_1, coordinates_2):
        x_difference = coord1[0] - coord2[0]
        y_difference = coord1[1] - coord2[1]
        distance = math.hypot(x_difference, y_difference)
        all_distances += distance
    return all_distances / len(coordinates_1)


def calculate_speed_in_kmps(image_1, image_2):
    time_difference = get_time_difference(image_1, image_2)

    image_1_cv, image_2_cv = convert_to_cv(image_1, image_2)

    key_points_1, key_points_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 750)

    matches = calculate_matches(descriptors_1, descriptors_2)

    coordinates_1, coordinates_2 = find_matching_coordinates(key_points_1, key_points_2, matches)

    distances = [math.hypot(coord1[0] - coord2[0], coord1[1] - coord2[1])
                 for coord1, coord2 in zip(coordinates_1, coordinates_2)]
    total_distance = sum(distances)
    average_distance = total_distance / len(distances)
    distance = average_distance * GSD / 100000
    speed = distance / time_difference
    return speed


def capture_image(file_path):
    with PiCamera() as camera:
        camera.resolution = (1920, 1080)
        camera.capture(file_path)
        time.sleep(1)


def calculate_and_average_speed(image_prefix, num_images):
    speeds = []

    for i in range(1, num_images):
        image_1 = f"{image_prefix}{i}.jpg"
        image_2 = f"{image_prefix}{i + 1}.jpg"

        speed = calculate_speed_in_kmps(image_1, image_2)
        speeds.append(speed)

    average_speed = sum(speeds) / len(speeds)
    return average_speed


def main():
    num_images = 100
    image_prefix = 'photo'

    start_time = time.time()
    end_time = start_time + 25 * 60  # Run for 25 minutes

    test_results = []

    while time.time() < end_time:
        for i in range(1, num_images + 1):
            image_path = f"{image_prefix}{i}.jpg"
            capture_image(image_path)

        average_speed = calculate_and_average_speed(image_prefix, num_images)
        test_results.append(average_speed)

    overall_average_speed = sum(test_results) / len(test_results)

    with open("result.txt", "w") as result_file:
        result_file.write(f"{overall_average_speed:.5f}")


if __name__ == "__main__":
    main()

