import fcntl
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
THRESHOLD = 0.45
LOG_FILE = "face.log"
COREML_COMPUTE_UNITS = "ALL"
COREML_PROVIDER = "CoreMLExecutionProvider"
BASE_DIR = Path(__file__).resolve().parent
ANTI_SPOOF_MODELS = (
    (BASE_DIR / "anti_spoof_models" / "MiniFASNetV1SE.onnx", 4.0),
    (BASE_DIR / "anti_spoof_models" / "MiniFASNetV2.onnx", 2.7),
)
PASSIVE_WINDOW_SIZE = 7
PASSIVE_REQUIRED_FRAMES = 3
PASSIVE_REAL_THRESHOLD = 0.60
PASSIVE_CAUTION_THRESHOLD = 0.50
PASSIVE_BLOCK_THRESHOLD = 0.35
PASSIVE_FAILURES_TO_BLOCK = 3
EYE_CALIBRATION_FRAMES = 8
EYE_CLOSE_RATIO = 0.80
EYE_REOPEN_RATIO = 0.88
MIN_EYE_DROP = 0.008
MIN_CLOSED_FRAMES = 1
HEAD_TURN_DEGREES = 14.0
CHALLENGE_TIMEOUT = 12.0
TRACK_TIMEOUT_SECONDS = 2.5
TRACK_EMBEDDING_THRESHOLD = 0.78
LIVE_TRACK_EMBEDDING_THRESHOLD = 0.80
TRACK_IOU_THRESHOLD = 0.10
TRACK_CENTER_THRESHOLD = 0.80
@dataclass
class LivenessState:
    stage: str = "blink"
    started_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    open_seen: bool = False
    closed_frames: int = 0
    eye_samples: list = field(default_factory=list)
    baseline_eye_ratio: float = 0.0
    baseline_yaw: float = 0.0
@dataclass
class FaceTrack:
    name: str
    bbox: np.ndarray
    embedding: np.ndarray
    last_seen_at: float
    spoof_scores: list = field(default_factory=list)
    spoof_failures: int = 0
class AntiSpoofEngine:
    def __init__(self):
        self.models = []
        for model_path, scale in ANTI_SPOOF_MODELS:
            if not model_path.exists():
                raise RuntimeError(f"缺少防伪模型: {model_path.name}")
            session = ort.InferenceSession(
                str(model_path),
                providers=[COREML_PROVIDER],
                provider_options=[
                    {"MLComputeUnits": COREML_COMPUTE_UNITS}
                ],
            )
            if COREML_PROVIDER not in session.get_providers():
                raise RuntimeError(
                    f"防伪模型 {model_path.name} 未启用 Core ML"
                )
            self.models.append(
                (
                    session,
                    session.get_inputs()[0].name,
                    session.get_outputs()[0].name,
                    scale,
                )
            )
    def crop_face(self, image, bbox, scale):
        image_height, image_width = image.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in bbox]
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        scale = min(
            (image_height - 1) / box_height,
            (image_width - 1) / box_width,
            scale,
        )
        center_x = x1 + box_width / 2
        center_y = y1 + box_height / 2
        crop_width = box_width * scale
        crop_height = box_height * scale
        crop_x1 = max(0, int(center_x - crop_width / 2))
        crop_y1 = max(0, int(center_y - crop_height / 2))
        crop_x2 = min(
            image_width - 1,
            int(center_x + crop_width / 2),
        )
        crop_y2 = min(
            image_height - 1,
            int(center_y + crop_height / 2),
        )
        crop = image[crop_y1:crop_y2 + 1, crop_x1:crop_x2 + 1]
        crop = cv2.resize(crop, (80, 80)).astype(np.float32)
        crop = np.transpose(crop, (2, 0, 1))
        return np.expand_dims(crop, axis=0)
    def predict(self, image, bbox):
        probabilities = []
        for session, input_name, output_name, scale in self.models:
            input_tensor = self.crop_face(image, bbox, scale)
            logits = session.run(
                [output_name],
                {input_name: input_tensor},
            )[0]
            exponentials = np.exp(
                logits - np.max(logits, axis=1, keepdims=True)
            )
            probabilities.append(
                exponentials / exponentials.sum(axis=1, keepdims=True)
            )
        combined = np.mean(probabilities, axis=0)
        return float(combined[0, 1])
def update_passive_liveness(track, real_score):
    track.spoof_scores.append(real_score)
    if len(track.spoof_scores) > PASSIVE_WINDOW_SIZE:
        del track.spoof_scores[0]
    median_score = float(np.median(track.spoof_scores))
    if median_score < PASSIVE_BLOCK_THRESHOLD:
        track.spoof_failures += 1
    else:
        track.spoof_failures = max(0, track.spoof_failures - 1)
    ready = len(track.spoof_scores) >= PASSIVE_REQUIRED_FRAMES
    trusted = ready and median_score >= PASSIVE_REAL_THRESHOLD
    blocked = track.spoof_failures >= PASSIVE_FAILURES_TO_BLOCK
    return trusted, blocked, median_score
def create_face_app():
    available_providers = ort.get_available_providers()
    if COREML_PROVIDER not in available_providers:
        raise RuntimeError(
            "没有可用的 Core ML 后端，程序拒绝使用纯 CPU 运行"
        )
    app = FaceAnalysis(
        name="buffalo_l",
        providers=[COREML_PROVIDER],
        provider_options=[
            {"MLComputeUnits": COREML_COMPUTE_UNITS}
        ],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    for task_name, model in app.models.items():
        active_providers = model.session.get_providers()
        if COREML_PROVIDER not in active_providers:
            raise RuntimeError(
                f"模型 {task_name} 未启用 Core ML，程序已停止"
            )
    print("硬件加速已启用: Core ML (Apple GPU / Neural Engine)")
    return app
def load_face_database():
    face_database = {}
    for file in Path(".").glob("*.npy"):
        embedding = np.load(file)
        norm = np.linalg.norm(embedding)
        if embedding.shape != (512,) or norm == 0:
            print(f"忽略无效人脸文件: {file.name}")
            continue
        face_database[file.stem] = embedding / norm
    if not face_database:
        raise RuntimeError("没有找到有效的 .npy 人脸文件")
    print("已加载的人脸：")
    for name in face_database:
        print(" -", name)
    return face_database
def write_logs(records):
    if not records:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = "".join(
        f"{now} | {name} | similarity={similarity:.3f}\n"
        for name, similarity in records
    )
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        try:
            file.write(lines)
            file.flush()
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
    for name, similarity in records:
        print(f"[LOG] {now} | {name} | similarity={similarity:.3f}")
def eye_aspect_ratio(eye):
    vertical_1 = np.linalg.norm(eye[1] - eye[5])
    vertical_2 = np.linalg.norm(eye[2] - eye[4])
    horizontal = np.linalg.norm(eye[0] - eye[3])
    if horizontal <= 1e-6:
        return 0.0
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))
def get_eye_ratio(face):
    landmarks = face.landmark_3d_68
    if landmarks is None or len(landmarks) < 48:
        return None
    points = np.asarray(landmarks[:, :2], dtype=np.float32)
    left_eye = points[36:42]
    right_eye = points[42:48]
    return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
def get_yaw_degrees(face):
    if face.pose is None or len(face.pose) < 2:
        return None
    return math.degrees(float(face.pose[1]))
def reset_liveness_state(state, now):
    state.stage = "blink"
    state.started_at = now
    state.last_seen_at = now
    state.open_seen = False
    state.closed_frames = 0
    state.eye_samples.clear()
    state.baseline_eye_ratio = 0.0
    state.baseline_yaw = 0.0
def check_liveness(name, face, now, states):
    state = states.get(name)
    if state is None:
        state = LivenessState(started_at=now, last_seen_at=now)
        states[name] = state
    state.last_seen_at = now
    if state.stage == "live":
        return True, "LIVE"
    if now - state.started_at > CHALLENGE_TIMEOUT:
        reset_liveness_state(state, now)
    eye_ratio = get_eye_ratio(face)
    yaw = get_yaw_degrees(face)
    if eye_ratio is None or yaw is None:
        return False, "HOLD STILL"
    if state.stage == "blink":
        if len(state.eye_samples) < EYE_CALIBRATION_FRAMES:
            state.eye_samples.append(eye_ratio)
            if len(state.eye_samples) == EYE_CALIBRATION_FRAMES:
                state.baseline_eye_ratio = float(
                    np.percentile(state.eye_samples, 80)
                )
                state.open_seen = True
            return False, "LOOK CAMERA"
        if eye_ratio > state.baseline_eye_ratio:
            state.baseline_eye_ratio = (
                state.baseline_eye_ratio * 0.9 + eye_ratio * 0.1
            )
        close_threshold = min(
            state.baseline_eye_ratio * EYE_CLOSE_RATIO,
            state.baseline_eye_ratio - MIN_EYE_DROP,
        )
        reopen_threshold = state.baseline_eye_ratio * EYE_REOPEN_RATIO
        if eye_ratio <= close_threshold:
            state.closed_frames += 1
            return False, "BLINK"
        if eye_ratio >= reopen_threshold:
            if state.closed_frames >= MIN_CLOSED_FRAMES:
                state.stage = "turn"
                state.started_at = now
                state.baseline_yaw = yaw
                return False, "TURN HEAD"
            state.closed_frames = 0
        return False, "BLINK"
    if state.stage == "turn":
        if abs(yaw - state.baseline_yaw) >= HEAD_TURN_DEGREES:
            state.stage = "live"
            return True, "LIVE"
        return False, "TURN HEAD"
    reset_liveness_state(state, now)
    return False, "BLINK"
def find_best_match(face, face_database):
    embedding = face.embedding
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return "UNKNOWN", -1.0, None
    embedding = embedding / norm
    best_name = "UNKNOWN"
    best_similarity = -1.0
    for name, known_embedding in face_database.items():
        similarity = float(np.dot(known_embedding, embedding))
        if similarity > best_similarity:
            best_similarity = similarity
            best_name = name
    return best_name, best_similarity, embedding
def bbox_iou(first, second):
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0,
        float(first[3] - first[1]),
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0,
        float(second[3] - second[1]),
    )
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union
def center_distance(first, second):
    first_center = np.array(
        [(first[0] + first[2]) / 2, (first[1] + first[3]) / 2]
    )
    second_center = np.array(
        [(second[0] + second[2]) / 2, (second[1] + second[3]) / 2]
    )
    scale = max(
        float(first[2] - first[0]),
        float(first[3] - first[1]),
        1.0,
    )
    return float(np.linalg.norm(first_center - second_center) / scale)
def assign_face_track(
    name,
    bbox,
    embedding,
    now,
    tracks,
    liveness_states,
    used_track_ids,
    next_track_id,
):
    selected_track_id = None
    selected_score = -1.0
    for track_id, track in tracks.items():
        if track_id in used_track_ids or track.name != name:
            continue
        if now - track.last_seen_at > TRACK_TIMEOUT_SECONDS:
            continue
        similarity = float(np.dot(track.embedding, embedding))
        state = liveness_states.get(track_id)
        minimum_similarity = TRACK_EMBEDDING_THRESHOLD
        if state is not None and state.stage == "live":
            minimum_similarity = LIVE_TRACK_EMBEDDING_THRESHOLD
        overlap = bbox_iou(track.bbox, bbox)
        distance = center_distance(track.bbox, bbox)
        spatial_match = (
            overlap >= TRACK_IOU_THRESHOLD
            or distance <= TRACK_CENTER_THRESHOLD
        )
        if similarity < minimum_similarity or not spatial_match:
            continue
        score = similarity + overlap - distance
        if score > selected_score:
            selected_score = score
            selected_track_id = track_id
    if selected_track_id is None:
        selected_track_id = next_track_id
        next_track_id += 1
        tracks[selected_track_id] = FaceTrack(
            name=name,
            bbox=np.asarray(bbox, dtype=np.float32).copy(),
            embedding=embedding.copy(),
            last_seen_at=now,
        )
    else:
        track = tracks[selected_track_id]
        track.bbox = np.asarray(bbox, dtype=np.float32).copy()
        track.embedding = embedding.copy()
        track.last_seen_at = now
    used_track_ids.add(selected_track_id)
    return selected_track_id, next_track_id
def draw_face_result(frame, face, label, color):
    x1, y1, x2, y2 = face.bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
def main():
    app = create_face_app()
    anti_spoof_engine = AntiSpoofEngine()
    face_database = load_face_database()
    liveness_states = {}
    tracks = {}
    next_track_id = 1
    last_log_times = {}
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")
    print()
    print("摄像头已启动")
    print("识别前必须完成眨眼和转头活体检测")
    print("支持同时识别多张人脸")
    print("按 Q 退出")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头")
                break
            now = time.time()
            faces = app.get(frame)
            live_faces = {}
            used_track_ids = set()
            for face in faces:
                best_name, best_similarity, embedding = find_best_match(
                    face,
                    face_database,
                )
                if best_similarity < THRESHOLD:
                    draw_face_result(
                        frame,
                        face,
                        f"UNKNOWN {best_similarity:.3f}",
                        (0, 0, 255),
                    )
                    continue
                track_id, next_track_id = assign_face_track(
                    best_name,
                    face.bbox,
                    embedding,
                    now,
                    tracks,
                    liveness_states,
                    used_track_ids,
                    next_track_id,
                )
                track = tracks[track_id]
                passive_score = anti_spoof_engine.predict(
                    frame,
                    face.bbox,
                )
                passive_trusted, passive_blocked, median_score = (
                    update_passive_liveness(track, passive_score)
                )
                state = liveness_states.get(track_id)
                already_live = state is not None and state.stage == "live"
                if passive_blocked:
                    if state is not None:
                        reset_liveness_state(state, now)
                    draw_face_result(
                        frame,
                        face,
                        f"SPOOF BLOCKED {median_score:.2f}",
                        (0, 0, 255),
                    )
                    continue
                if passive_score < PASSIVE_CAUTION_THRESHOLD:
                    draw_face_result(
                        frame,
                        face,
                        f"VERIFY: CHECKING {passive_score:.2f}",
                        (0, 165, 255),
                    )
                    continue
                if not passive_trusted and not already_live:
                    draw_face_result(
                        frame,
                        face,
                        f"VERIFY: CHECKING {median_score:.2f}",
                        (0, 165, 255),
                    )
                    continue
                is_live, prompt = check_liveness(
                    track_id,
                    face,
                    now,
                    liveness_states,
                )
                if not is_live:
                    draw_face_result(
                        frame,
                        face,
                        f"VERIFY: {prompt}",
                        (0, 165, 255),
                    )
                    continue
                previous_similarity = live_faces.get(best_name, -1.0)
                live_faces[best_name] = max(
                    previous_similarity,
                    best_similarity,
                )
                draw_face_result(
                    frame,
                    face,
                    f"{best_name} {best_similarity:.3f} LIVE",
                    (0, 255, 0),
                )
            for track_id in list(tracks):
                track = tracks[track_id]
                if (
                    track_id not in used_track_ids
                    and now - track.last_seen_at > TRACK_TIMEOUT_SECONDS
                ):
                    del tracks[track_id]
                    liveness_states.pop(track_id, None)
            log_is_due = any(
                now - last_log_times.get(name, 0) >= 1
                for name in live_faces
            )
            if live_faces and log_is_due:
                records = list(live_faces.items())
                write_logs(records)
                for name in live_faces:
                    last_log_times[name] = now
            cv2.imshow("InsightFace Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
