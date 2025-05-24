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


import os
import cv2
import faiss
import numpy as np
from PIL import Image
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from collections import Counter
from langchain_community.vectorstores import FAISS
from faiss_CRUD import DummyEmbeddings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DetectionProcessingService:
    # === Configure ===
    base_dir = os.getcwd()
    YOLO_MODEL = "yolov11n-face.pt"
    FACE_EMBEDDER_MODEL = "vggface2"
    yolo_model_path = os.path.join(base_dir, "model", YOLO_MODEL)
    known_faces_path = os.path.join(base_dir, "data", "faiss_Store")
    index_path = os.path.join(known_faces_path, "index.faiss")

    model_YOLO = YOLO(yolo_model_path)
    model_Facenet = InceptionResnetV1(pretrained=FACE_EMBEDDER_MODEL).eval()

    # === Confident ===
    YOLO_THRESHOLD = 0.75
    FACENET_THRESHOLD = 0.65

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

    def load_faiss_index(self):
        """
        โหลดฐานข้อมูล FAISS และสร้าง mapping ID → Name
        """
        try:
            self.faiss_db = FAISS.load_local(
                folder_path=self.known_faces_path,
                embeddings=DummyEmbeddings(),
                allow_dangerous_deserialization=True,
            )
            self.index = self.faiss_db.index
            if not isinstance(self.index, faiss.IndexIDMap):
                raise ValueError("Loaded index must be IndexIDMap")
            self.index_ivf = self.index
            docstore_dict = self.faiss_db.docstore._dict
            index_to_docstore_id = self.faiss_db.index_to_docstore_id
            self.id_to_name = {
                faiss_idx: docstore_dict[doc_id].metadata.get("name", "Unknown")
                for faiss_idx, doc_id in index_to_docstore_id.items()
                if doc_id in docstore_dict
            }
            print("[✅] FAISS index and mappings loaded successfully.")
        except Exception as e:
            print(f"[❗] Error loading FAISS index: {str(e)}")
            raise

    def find_best_match(self, embedding):
        """Find best match from FAISS index"""
        if self.index is None:
            raise ValueError("FAISS index not loaded.")
        embedding = embedding / np.linalg.norm(embedding)
        distances, indices = self.index_ivf.search(np.array([embedding]), k=3)
        matched_names = [self.id_to_name.get(int(i), "Unknown") for i in indices[0]]
        name_counter = Counter(matched_names)
        most_common_name = (
            name_counter.most_common(1)[0][0] if name_counter else "Unknow"
        )
        if distances[0][0] < self.FACENET_THRESHOLD:
            return most_common_name
        else:
            return "Unknow"

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


def process_single_image(image_path, service):
    service.load_faiss_index()
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
        person = service.find_best_match(embedding)  # ใช้ instance
        print(f"พบใบหน้าที่ตำแหน่ง {pos} - ชื่อ: {person}")
        x, y = pos
        text = f"{person}"
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
    service = DetectionProcessingService()
    image_path = "D:/Project/FaceScan/data/person_img/P/494816178_1415928936275008_922969240883063143_n.jpg"  # ระบุไฟล์ภาพ
    process_single_image(image_path, service)
