import os
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
import pillow_heif
import uuid
import faiss

# ✅ เปลี่ยนเป็น langchain_community ตาม v0.2+
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from faiss import IndexFlatL2
import os

# Get the current working directory
current_path = os.getcwd()
print("Current Working Directory:", current_path)

pillow_heif.register_heif_opener()


class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return texts

    def embed_query(self, text):
        return text


class FaceVectorDatabase:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = 512
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.face_detector = YOLO(
            os.path.join(f"{current_path}", "model/yolov11n-face.pt")
        )
        self.faiss_storage_path = os.path.join(f"{current_path}", "data/faiss_Store")
        self.data_path = os.path.join(f"{current_path}", "data/person_img")
        self.update_path = os.path.join(f"{current_path}", "data/person_img_update")
        self.transform = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def extract_face_vectors(self, image_folder):
        """
        image_folder (image_person / image_person_update) : folder ที่ต้องการจะเข้าไปเอาภาพจาก folder ที่เรียงแบบ

        ├── person_img/
        │ ├── name1/
        │ │ ├── 1.jpg
        │ │ ├── 2.jpg
        │ │ └── 3.jpg
        │ ├── name2/
        │ │ ├── 1.jpg
        │ │ ├── 2.jpg
        │ │ └── 3.jpg
        │ └── ...

        vectors : ใช้เก็บ vector ที่ถูกบีบอัดแล้ว
        doc     :
        """

        vectors = []
        docs = []

        for folder_person_name in tqdm(
            os.listdir(image_folder), desc=f"Processing {image_folder}"
        ):
            # path ที่เก็บ folder ของแต่ละคน เช่น Pun ข้างในมีภาพ
            person_folder = os.path.join(image_folder, folder_person_name)
            # ถ้าไม่มีภาพใน folder ก็ไปหา folder ถัดไป
            if not os.path.isdir(person_folder):
                continue
            # ถ้ามีเอาภาพมาบีบ vector
            for img_file in os.listdir(person_folder):
                # เข้าถึงภาพใน folder ของคนนั้นโดยการเข้าถึงผ่าน path person_folder
                img_path = os.path.join(person_folder, img_file)

                try:
                    img = Image.open(img_path).convert("RGB")  # แปลงให้เป้น RGB ก่อน
                except Exception as e:
                    print(f"[!] Error loading image: {img_path} | {e}")
                    continue

                """
                YOLO FUNCTION DETECT FACE AND CROPPED FACE
                """

                # Detect face YOLO
                results = self.face_detector.predict(
                    source=img, conf=0.8, verbose=False
                )
                detections = results[0]

                if not detections.boxes or len(detections.boxes) == 0:
                    print(f"[!] No face found in: {img_path}")
                    continue

                x1, y1, x2, y2 = map(int, detections.boxes[0].xyxy[0])
                cropped = img.crop((x1, y1, x2, y2))
                face_tensor = self.transform(cropped).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    embedding = self.model(face_tensor).squeeze().cpu().numpy()
                    embedding = embedding / np.linalg.norm(embedding)  # Normalize
                    # เก็บ vector ที่ถูกบีบอัดไว้ใน list vector,doc
                    vectors.append(embedding)
                    docs.append(
                        Document(
                            page_content="face_vector",
                            metadata={"name": folder_person_name, "image": img_file},
                        )
                    )
        return vectors, docs

    """
    ================================CREATE================================
    """

    def create_empty_faiss(self):
        """
        index    : สร้าง vector ขนาด 512
        docstore : สร้าง ที่ว่าง ๆ ที่เก็บ metadata เช่น ชื่อบุคคลและชื่อไฟล์
        index_to_docstore_id : dictionary mapping จาก index ใน FAISS ไปยัง ID ใน docstore

        db
        index: FAISS index สำหรับจัดเก็บเวกเตอร์
        docstore: เก็บ metadata เช่น ชื่อคนและชื่อรูป
        index_to_docstore_id: mapping ระหว่าง index กับ ID ใน docstore
        """

        index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dim))  # ใช้ IndexIDMap
        docstore = InMemoryDocstore({})
        index_to_docstore_id = {}
        db = FAISS(
            embedding_function=DummyEmbeddings(),
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )
        db.save_local(self.faiss_storage_path)
        print("[✅] Empty FAISS database created and saved.")

    def build_faiss(self, batch_size=5):
        """
        batch_size : แบ่งเป็น batch เพื่อลดการใช้ RAM
        index : สร้าง IndexFlatL2 -ขนาด 512
        index.add : เอา vector ที่ extract มาไปใส่ ใน index
        docstore : loop เอา vector ที่อยู่ใน index มาเก็บใน docstore

        doc_ids : เก็บ uuid ตามจำนวน vector ที่ได้จาก image_person
        docstore_dict : เป็น unique id ของแต่ละภาพ
        index_to_docstore_id : สสร้าง dictionary ที่ mapping ระหว่าง ลำดับ index ใน FAISS กับ ID ของเอกสารใน docstore
        """
        vectors, docs = self.extract_face_vectors(self.data_path)
        if not vectors:
            print("[❗] No face vectors extracted.")
            return

        index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dim))
        ids = np.array(range(len(vectors)), dtype=np.int64)  # กำหนด ID แบบเรียงลำดับ
        for i in range(0, len(vectors), batch_size):
            batch_vecs = np.array(vectors[i : i + batch_size]).astype(np.float32)
            batch_ids = ids[i : i + batch_size]
            index.add_with_ids(batch_vecs, batch_ids)

        docstore_dict = {}
        index_to_docstore_id = {}
        for i, doc in enumerate(docs):
            doc_id = str(uuid.uuid4())
            docstore_dict[doc_id] = doc
            index_to_docstore_id[i] = doc_id  # mapping FAISS idx -> doc_id

        db = FAISS(
            embedding_function=DummyEmbeddings(),
            index=index,
            docstore=InMemoryDocstore(docstore_dict),
            index_to_docstore_id=index_to_docstore_id,
        )
        db.save_local(self.faiss_storage_path)
        print("[✅] FAISS database built with IndexIDMap and saved.")

    """
    ================================UPDATE================================
    """

    def update_faiss(self):
        """
        db
        load model vector : เดิมมาใช้งานเป็น base

        existing_keys : เช็คว่า Metadata={'name': 'Pun', 'image': 'IMG_6371.HEIC'}
        ซ้ำกับชื่อและภาพที่อยู่ใน model มั้ย


        """
        db = FAISS.load_local(
            self.faiss_storage_path,
            embeddings=DummyEmbeddings(),
            allow_dangerous_deserialization=True,
        )
        existing_keys = {
            f"{doc.metadata['name']}_{doc.metadata['image']}"
            for doc in db.docstore._dict.values()
        }

        new_vectors, new_docs = self.extract_face_vectors(self.update_path)
        if not new_vectors:
            print("[❗] No new faces to add.")
            return

        filtered_vectors = []
        filtered_docs = []
        for vec, doc in zip(new_vectors, new_docs):
            key = f"{doc.metadata['name']}_{doc.metadata['image']}"
            if key not in existing_keys:
                filtered_vectors.append(vec)
                filtered_docs.append(doc)
                existing_keys.add(key)

        if not filtered_vectors:
            print("[ℹ️] All vectors already exist in FAISS.")
            return

        current_count = db.index.ntotal
        ids = np.array(
            range(current_count, current_count + len(filtered_vectors)), dtype=np.int64
        )
        db.index.add_with_ids(np.array(filtered_vectors).astype(np.float32), ids)

        # อัปเดต docstore และ index_to_docstore_id
        for i, (vec, doc) in enumerate(zip(filtered_vectors, filtered_docs)):
            doc_id = str(uuid.uuid4())
            db.docstore._dict[doc_id] = doc
            db.index_to_docstore_id[current_count + i] = doc_id

        db.save_local(self.faiss_storage_path)

    """
    ================================DELETE================================
    """

    def delete_vectors_by_name(self, name_to_delete):
        """
        เช็คจากชื่อ ว่าใน metadata มีชื่อนั้นมั้ย ถ้ามี ลบ
        """
        db = FAISS.load_local(
            self.faiss_storage_path,
            embeddings=DummyEmbeddings(),
            allow_dangerous_deserialization=True,
        )

        indices_to_delete = []
        for idx, doc_id in db.index_to_docstore_id.items():
            doc = db.docstore.search(doc_id)
            if doc.metadata.get("name") == name_to_delete:
                faiss_id = idx  # เพราะตอนนี้ idx คือ FAISS ID จริงๆ
                indices_to_delete.append(faiss_id)

        if not indices_to_delete:
            print(f"[❗] No vectors found with name '{name_to_delete}'.")
            return

        print(
            f"[🗑️] Deleting {len(indices_to_delete)} vectors for '{name_to_delete}'..."
        )
        db.index.remove_ids(np.array(indices_to_delete, dtype=np.int64))

        # อัปเดต docstore และ index_to_docstore_id
        remaining_index_map = {}
        remaining_docstore = {}
        for idx, doc_id in db.index_to_docstore_id.items():
            if idx not in indices_to_delete:
                remaining_index_map[idx] = doc_id
                remaining_docstore[doc_id] = db.docstore.search(doc_id)

        db.docstore = InMemoryDocstore(remaining_docstore)
        db.index_to_docstore_id = remaining_index_map

        db.save_local(self.faiss_storage_path)

    """
    ================================GET================================
    """

    def get_total_face_count(self):
        """
        นับจำนวนเวกเตอร์ใบหน้าทั้งหมดใน FAISS database
        """
        db = FAISS.load_local(
            self.faiss_storage_path,
            embeddings=DummyEmbeddings(),
            allow_dangerous_deserialization=True,
        )
        total_count = db.index.ntotal
        print(f"[📊] Total face vectors in FAISS: {total_count}")
        return total_count

    def get_person_vectors(self, person_name):
        """
        database มี person_name(ชื่อนั้นกี่ภาพและอะไรบ้าง)
        """
        db = FAISS.load_local(
            self.faiss_storage_path,
            embeddings=DummyEmbeddings(),
            allow_dangerous_deserialization=True,
        )
        results = []
        for idx, doc_id in db.index_to_docstore_id.items():
            doc = db.docstore.search(doc_id)
            if doc.metadata.get("name") == person_name:
                result_info = {"index": idx, "doc_id": doc_id, "metadata": doc.metadata}
                results.append(result_info)
                print(
                    f"[🔍] Found vector: Index={idx}, Doc ID='{doc_id}', Metadata={doc.metadata}"
                )
        print(f"[ℹ️] Total vectors found for '{person_name}': {len(results)}")
        return results


if __name__ == "__main__":
    db_manager = FaceVectorDatabase()

    # สร้าง FAISS database empty รอ add หน้าใหม่
    # db_manager.create_empty_faiss()

    # สร้าง FAISS database จากภาพใน person_img/
    db_manager.build_faiss()

    # อัปเดตฐานข้อมูลจาก person_img_update/
    # db_manager.update_faiss()

    # ลบเวกเตอร์ของบุคคล
    # db_manager.delete_vectors_by_name("Pun")

    # ค้นหาเวกเตอร์ของบุคคลหนึ่ง
    # db_manager.get_total_face_count() #ทั้งหมดมีกี่คน
    # db_manager.get_person_vectors("Pun") คนเดียว
