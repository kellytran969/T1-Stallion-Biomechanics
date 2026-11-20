import math
import pandas as pd
import csv

KEYPOINT_MAP = {
    "4": "poll",
    "7": "withers",
    "10": "croup",
    "18": "pelvis",
    "22": "hind_hoof",
    "11": "chest",
    "D1": "dummy_d1",
    "D2": "dummy_d2"
}

def distance(point1, point2):
    """
    Calculate the distance between two points.
    Each point is represented as (x, y).
    """
    x1, y1 = point1
    x2, y2 = point2

    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def horizontal_distance(point1, point2):
    """
    Calculate the horizontal distance between two points.
    """
    return point2[0] - point1[0]

def angle_from_horizontal(point1, point2):
    """
    Calculate the angle of a line relative to horizontal.
    Returns the angle in degrees.
    """
    dx = point2[0] - point1[0]
    dy = point1[1] - point2[1]

    return math.degrees(math.atan2(dy, dx))

def signed_angle(vector1, vector2):
    """
    Calculate the signed angle between two vectors.
    Returns the angle in degrees.
    """
    x1, y1 = vector1
    x2, y2 = vector2

    cross = x1 * y2 - y1 * x2
    dot = x1 * x2 + y1 * y2

    return math.degrees(math.atan2(cross, dot))

def get_horse_direction(poll, croup):
    """
    Determine which direction the horse is facing.
    Returns:
        -1 if facing left
         1 if facing right
    """
    if poll[0] < croup[0]:
        return -1
    elif poll[0] > croup[0]:
        return 1
    else:
        return None

def calculate_endpoints(landmarks, direction=None):
    """
    Calculate the four biomechanics endpoints for one frame.
    landmarks: dictionary containing landmark coordinates.
    direction:
        -1 = horse facing left
         1 = horse facing right
        None = determine direction automatically.
    Missing or invalid measurements return None.
    """

    # Get the landmarks we need
    poll = landmarks.get("poll")
    withers = landmarks.get("withers")
    croup = landmarks.get("croup")
    pelvis = landmarks.get("pelvis")
    hind_hoof = landmarks.get("hind_hoof")
    chest = landmarks.get("chest")
    dummy_d1 = landmarks.get("dummy_d1")
    dummy_d2 = landmarks.get("dummy_d2")

    # Determine direction if it was not provided
    if direction is None and poll is not None and croup is not None:
        direction = get_horse_direction(poll, croup)

    # Check that direction is valid
    if direction not in (-1, 1):
        direction = None

    results = {
        "back_angle": None,
        "neck_angle": None,
        "hind_position": None,
        "chest_position": None
    }

    # 1. BACK ANGLE
    # Positive = croup above withers
    if withers is not None and croup is not None:
        dx = abs(croup[0] - withers[0])
        dy = withers[1] - croup[1]

        if dx != 0:
            results["back_angle"] = math.degrees(math.atan2(dy, dx))

    # 2. NECK ANGLE
    # Positive = poll above the back-line direction
    if direction is not None and poll is not None and withers is not None and croup is not None:

        back_vector = (
            withers[0] - croup[0],
            croup[1] - withers[1]
        )

        neck_vector = (
            poll[0] - withers[0],
            withers[1] - poll[1]
        )

        results["neck_angle"] = (signed_angle(back_vector, neck_vector) * direction)

    # Calculate torso length for normalization
    torso_length = None

    if withers is not None and croup is not None:
        torso_length = distance(withers, croup)

    # 3. HIND-LIMB POSITION
    # Positive = hoof forward of pelvis
    if direction is not None and pelvis is not None and hind_hoof is not None and torso_length is not None and torso_length != 0:

        horizontal_offset = (hind_hoof[0] - pelvis[0]) * direction

        results["hind_position"] = (horizontal_offset / torso_length)

    # 4. CHEST POSITION
    # Positive = chest toward the dummy
    if direction is not None and chest is not None and dummy_d1 is not None and dummy_d2 is not None:

        dummy_length = distance(dummy_d1, dummy_d2)

        if dummy_length != 0:
            horizontal_offset = (dummy_d1[0] - chest[0]) * direction
            results["chest_position"] = horizontal_offset / dummy_length

    return results

def process_tracking_csv(input_file, output_file):
    """
    Read landmark coordinates from a CSV,
    calculate endpoints for every frame,
    and save the results to a new CSV.

    Expected columns:
        frame, keypoint, x, y, visible
    """

    # Read the tracking data
    tracking_data = pd.read_csv(input_file, dtype={"keypoint": str})

    # Check that required columns exist
    required_columns = {"frame", "keypoint", "x", "y", "visible"}

    if not required_columns.issubset(tracking_data.columns):
        raise ValueError(
            "CSV must contain: frame, keypoint, x, y, visible"
        )

    all_results = []

    # Process one frame at a time
    for frame_number, frame_data in tracking_data.groupby("frame"):

        # Create a dictionary of landmark coordinates
        landmarks = {}

        for _, row in frame_data.iterrows():

            keypoint_id = row["keypoint"]

            # Convert keypoint ID into landmark name
            if keypoint_id not in KEYPOINT_MAP:
                continue

            landmark_name = KEYPOINT_MAP[keypoint_id]

            # Only use visible landmarks
            if row["visible"] == 1:
                landmarks[landmark_name] = (
                    row["x"],
                    row["y"]
                )
            else:
                landmarks[landmark_name] = None

        # Calculate the four endpoints
        results = calculate_endpoints(landmarks)

        # Add the frame number
        results["frame"] = frame_number

        all_results.append(results)

    # Convert results into a DataFrame
    results_df = pd.DataFrame(all_results)

    # Match the column names expected by overlay_demo.py
    results_df = results_df.rename(columns={
        "back_angle": "theta_back_deg",
        "neck_angle": "theta_neck_deg",
        "hind_position": "hind_norm",
        "chest_position": "chest_norm"
    })

    results_df = results_df[
        ["frame", "theta_back_deg", "theta_neck_deg", "hind_norm", "chest_norm"]
    ]

    # Save results
    results_df.to_csv(output_file, index=False)

    print("Endpoint calculations complete!")
    print("Saved results to:", output_file)
    print(results_df.head())

if __name__ == "__main__":
    process_tracking_csv(
        "test_tracking.csv",
        "test_endpoints.csv"
    )