import os
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
import pillow_heif

# ✅ เปลี่ยนเป็น langchain_community ตาม v0.2+
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain_community.docstore.in_memory import (
    InMemoryDocstore,
)  # ✅ ใช้จาก community
from faiss import IndexFlatL2


# ลงทะเบียน HEIF opener
pillow_heif.register_heif_opener()

# -------------------------------
# 🔧 Setup Model + Transform
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load embedding model
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Load face detection model
face_detector = YOLO("D:/Project/FaceDetect/model/yolov11n-face.pt")

# Transform image
transform = transforms.Compose(
    [
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


# -------------------------------
# 📁 Helper Function: Read Images and Extract Face Vectors
# -------------------------------
def extract_face_vectors(image_folder):
    vectors = []
    docs = []

    for person_name in tqdm(
        os.listdir(image_folder), desc=f"Processing {image_folder}"
    ):
        person_folder = os.path.join(image_folder, person_name)
        if not os.path.isdir(person_folder):
            continue

        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[!] Error loading image: {img_path} | {e}")
                continue

            # Detect face
            results = face_detector.predict(source=img, conf=0.8, verbose=False)
            detections = results[0]

            if not detections.boxes or len(detections.boxes) == 0:
                print(f"[!] No face found in: {img_path}")
                continue

            # Take first face
            x1, y1, x2, y2 = map(int, detections.boxes[0].xyxy[0])
            cropped = img.crop((x1, y1, x2, y2))
            face_tensor = transform(cropped).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model(face_tensor).squeeze().cpu().numpy()
                embedding = embedding / np.linalg.norm(embedding)  # Normalize

                vectors.append(embedding)
                docs.append(
                    Document(
                        page_content="face_vector",
                        metadata={"name": person_name, "image": img_file},
                    )
                )

    return vectors, docs


# -------------------------------
# 🧠 Class DummyEmbeddings (จำเป็นสำหรับ FAISS)
# -------------------------------
class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return texts

    def embed_query(self, text):
        return text


# -------------------------------
# 🏗️ 1. สร้าง FAISS จาก person_img/
# -------------------------------
def build_faiss():
    data_path = "D:/Project/FaceDetect/data/person_img"
    vectors, docs = extract_face_vectors(data_path)

    index = IndexFlatL2(vectors[0].shape[0])  # dim should match embedding size
    index.add(np.array(vectors).astype(np.float32))

    db = FAISS(
        embedding_function=DummyEmbeddings(),
        index=index,
        docstore=InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)}),
        index_to_docstore_id={i: str(i) for i in range(len(docs))},
    )

    # ✅ บันทึกใน path ที่ถูกต้อง
    db.save_local("D:/Project/FaceDetect/data/faiss_Store")
    print("[✅] FAISS database created and saved.")


# -------------------------------
# 🔄 2. อัปเดต FAISS จาก person_img_update/
# -------------------------------
def update_faiss():
    db = FAISS.load_local(
        "D:/Project/FaceDetect/data/faiss_Store",
        embeddings=DummyEmbeddings(),
        allow_dangerous_deserialization=True,
    )

    update_path = "D:/Project/FaceDetect/data/person_img_update"
    new_vectors, new_docs = extract_face_vectors(update_path)

    if not new_vectors:
        print("[❗] No new faces to add.")
        return

    # สร้าง text_embeddings
    text_embeddings = [("face_vector", vec.astype(np.float32)) for vec in new_vectors]

    # หา ID ต่อจาก index ปัจจุบัน
    current_index_length = db.index.ntotal
    ids = [str(i + current_index_length) for i in range(len(new_vectors))]

    # เพิ่มลง docstore
    for doc_id, doc in zip(ids, new_docs):
        db.docstore._dict[doc_id] = doc

    # เพิ่มลง FAISS index
    db.index.add(np.array([te[1] for te in text_embeddings]).astype(np.float32))

    # อัปเดต mapping ระหว่าง index -> docstore id
    db.index_to_docstore_id.update(
        {i + current_index_length: ids[i] for i in range(len(ids))}
    )

    # บันทึก
    db.save_local("D:/Project/FaceDetect/data/faiss_Store")
    print(f"[✅] Added {len(new_vectors)} new faces with metadata to FAISS.")


# -------------------------------
# ❌ 3. ลบ vector ที่มีชื่อบุคคลเฉพาะ
# -------------------------------
def delete_vectors_by_name(name_to_delete):
    db = FAISS.load_local(
        "D:/Project/FaceDetect/data/faiss_Store",
        embeddings=DummyEmbeddings(),
        allow_dangerous_deserialization=True,
    )

    indices_to_delete = []
    for idx in range(db.index.ntotal):
        docstore_id = db.index_to_docstore_id[idx]
        doc = db.docstore.search(docstore_id)
        if doc.metadata.get("name") == name_to_delete:
            indices_to_delete.append(idx)

    if not indices_to_delete:
        print(f"[❗] No vectors found with name '{name_to_delete}'.")
        return

    # กรอง vector
    all_vectors = db.index.reconstruct_n(0, db.index.ntotal)
    mask = np.ones(all_vectors.shape[0], dtype=bool)
    mask[indices_to_delete] = False
    new_vectors = all_vectors[mask]

    # สร้าง index ใหม่
    new_index = IndexFlatL2(new_vectors.shape[1])
    new_index.add(new_vectors.astype(np.float32))

    # กรอง docstore
    remaining_ids = [
        db.index_to_docstore_id[i]
        for i in range(db.index.ntotal)
        if i not in indices_to_delete
    ]
    new_docstore = InMemoryDocstore(
        {doc_id: db.docstore.search(doc_id) for doc_id in remaining_ids}
    )
    new_index_to_docstore_id = {i: doc_id for i, doc_id in enumerate(remaining_ids)}

    # สร้าง FAISS ใหม่
    new_db = FAISS(
        embedding_function=db.embedding_function,
        index=new_index,
        docstore=new_docstore,
        index_to_docstore_id=new_index_to_docstore_id,
    )

    new_db.save_local("D:/Project/FaceDetect/data/faiss_Store")
    print(
        f"[✅] Deleted {len(indices_to_delete)} vectors with name '{name_to_delete}'."
    )


# -------------------------------
# 🚀 Main CLI Menu
# -------------------------------
def main_menu():
    while True:
        print("\n=== FAISS Face Database Manager ===")
        print("1. Build FAISS from person_img")
        print("2. Update FAISS from img_update")
        print("3. Delete face by name")
        print("4. Exit")
        choice = input("Select option (1-4): ").strip()

        if choice == "1":
            build_faiss()
        elif choice == "2":
            update_faiss()
        elif choice == "3":
            name = input("Enter name to delete: ").strip()
            delete_vectors_by_name(name)
        elif choice == "4":
            print("[👋] Exiting program.")
            break
        else:
            print("[❗] Invalid option. Please try again.")

        print("-------------------------------")


if __name__ == "__main__":
    main_menu()
