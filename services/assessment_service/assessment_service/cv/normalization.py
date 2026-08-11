def normalize_landmark(point, origin, scale):
    if point is None or origin is None or not scale:
        return [0.0, 0.0, 0.0]
    return [
        (float(point.get("x", 0.0)) - origin[0]) / scale,
        (float(point.get("y", 0.0)) - origin[1]) / scale,
        (float(point.get("z", 0.0)) - origin[2]) / scale,
    ]


def body_origin_and_scale(pose_landmarks):
    left = pose_landmarks[11] if len(pose_landmarks) > 11 else None
    right = pose_landmarks[12] if len(pose_landmarks) > 12 else None
    if not left or not right:
        return [0.5, 0.5, 0.0], 0.33
    origin = [
        (float(left.get("x", 0.0)) + float(right.get("x", 0.0))) / 2,
        (float(left.get("y", 0.0)) + float(right.get("y", 0.0))) / 2,
        (float(left.get("z", 0.0)) + float(right.get("z", 0.0))) / 2,
    ]
    scale = ((float(left.get("x", 0.0)) - float(right.get("x", 0.0))) ** 2 + (float(left.get("y", 0.0)) - float(right.get("y", 0.0))) ** 2) ** 0.5
    return origin, scale or 0.33
