import os
import cv2
import faiss
import numpy as np
from PIL import Image
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from collections import Counter
from langchain.vectorstores import FAISS

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DetectionProcessingService:
    # === Configure ===
    base_dir = os.getcwd()
    YOLO_MODEL = "yolov11n-face.pt"  # ชื่อ model YOLO ยิ่งเล็กทำงานไว
    FACE_EMBEDDER_MODEL = "vggface2"
    yolo_model_path = os.path.join(base_dir, "model", YOLO_MODEL)
    known_faces_path = os.path.join(base_dir, "data", "faisss_Store")
    index_path = os.path.join(known_faces_path, "index.faiss")

    model_YOLO = YOLO(yolo_model_path)
    model_Facenet = InceptionResnetV1(pretrained=FACE_EMBEDDER_MODEL).eval()

    # === Confident ===
    YOLO_THRESHOLD = 0.75 #0.75-0.9 มากแม่น
    FACENET_THRESHOLD = 0.65 #0.6-0.75 น้อยแม่น

    transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    index = None
    index_ivf = None
    id_to_name = {}

    @classmethod
    def load_faiss_index(cls):
        """โหลด FAISS และสร้าง mapping ID → name"""
        cls.index = faiss.read_index(cls.index_path)
        dimension = cls.index.d
        new_index = faiss.IndexFlatL2(dimension)
        cls.index_ivf = faiss.IndexIDMap(new_index)
        vectors = cls.index.reconstruct_n(0, cls.index.ntotal)
        ids = np.array(range(cls.index.ntotal), dtype=np.int64)
        cls.index_ivf.add_with_ids(vectors, ids)

        cls.faiss_db = FAISS.load_local(
            cls.known_faces_path,
            embeddings=None,
            allow_dangerous_deserialization=True,
        )
        cls.id_to_name = {}
        docstore = cls.faiss_db.docstore._dict
        for idx, value in docstore.items():
            try:
                name = value.metadata.get("name", "Unknown")
                cls.id_to_name[int(idx)] = name
            except:
                continue

    @classmethod
    def detect_faces(cls, frame):
        """YOLO: Detect faces in the image"""
        results = cls.model_YOLO.predict(
            source=frame, conf=cls.YOLO_THRESHOLD, verbose=False
        )
        return results[0]

    @classmethod
    def extract_faces_and_positions(cls, frame, detections):
        """Extract face crops and their positions"""
        positions, faces = [], []
        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            faces.append(frame[y1:y2, x1:x2])
            positions.append(((x1 + x2) // 2, (y1 + y2) // 2))
        return positions, faces

    @classmethod
    def image_embedding(cls, cropped_image):
        """Convert face image to embedding vector"""
        pil_img = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        tensor = cls.transform(pil_img).unsqueeze(0)
        return cls.model_Facenet(tensor).detach().cpu().numpy()[0]

    @classmethod
    def find_best_match(cls, embedding):
        """Find best match from FAISS index"""
        if cls.index is None:
            raise ValueError("FAISS index not loaded.")

        embedding = embedding / np.linalg.norm(embedding)
        distances, indices = cls.index_ivf.search(np.array([embedding]), k=3)

        matched_names = [cls.id_to_name.get(int(i), "Unknown") for i in indices[0]]
        name_counter = Counter(matched_names)
        most_common_name = (
            name_counter.most_common(1)[0][0] if name_counter else "Unknow"
        )

        if distances[0][0] < cls.FACENET_THRESHOLD:
            return most_common_name
        else:
            return "Unknow"


def process_single_image(image_path):
    DetectionProcessingService.load_faiss_index()
    frame = cv2.imread(image_path)
    if frame is None:
        print("ไม่สามารถโหลดรูปภาพได้")
        return

    detections = DetectionProcessingService.detect_faces(frame)
    annotated_frame = detections.plot()

    if not detections.boxes:
        print("ไม่พบใบหน้าในภาพ")
        return

    positions, faces = DetectionProcessingService.extract_faces_and_positions(
        frame, detections
    )

    for pos, face in zip(positions, faces):
        embedding = DetectionProcessingService.image_embedding(face)
        person = DetectionProcessingService.find_best_match(embedding)

        print(f"พบใบหน้าที่ตำแหน่ง {pos} - ชื่อ: {person}")

        x, y = pos
        text = f"{person}"
        """
        Show ภาพที่ Label ไว้
        """
        cv2.putText(
            annotated_frame,
            text,
            (x - 40, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.circle(annotated_frame, (x, y), 5, (255, 0, 0), -1)

    cv2.imshow("Face Recognition Result", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # กำหนด path ของภาพที่ต้องการประมวลผล

    """
    path รูปภาพ
    """
    image_path = (
        "D:/Project/FaceScan/data/person_img/"  
    )
    process_single_image(image_path)
