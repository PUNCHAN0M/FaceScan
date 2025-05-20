# FAISS Face Recognition System

ระบบจัดการข้อมูลใบหน้าด้วย FAISS (Facebook AI Similarity Search) สำหรับงานจำแนกและค้นหาใบหน้าจากรูปภาพ

##โครงสร้างโปรเจกต์
```
project_root/
│
├── data/
│ └── faiss_Store/
│ ├── index.faiss # ไฟล์ FAISS สำหรับเก็บเวกเตอร์ของใบหน้า
│ └── index.pkl # ไฟล์ pickle เก็บ metadata เช่น mapping ของชื่อบุคคลกับเวกเตอร์
│
├── person_img/ # โฟลเดอร์รูปภาพสำหรับสร้าง FAISS (ใช้ครั้งแรกเท่านั้น)
│ ├── name1/
│ │ ├── 1.jpg
│ │ ├── 2.jpg
│ │ └── 3.jpg
│ ├── name2/
│ │ ├── 1.jpg
│ │ ├── 2.jpg
│ │ └── 3.jpg
│ └── ...
│
├── person_img_update/ # โฟลเดอร์รูปภาพสำหรับอัปเดต FAISS (เพิ่มคนใหม่หรือรูปใหม่)
│ ├── name1/
│ │ ├── 1.jpg
│ │ ├── 2.jpg
│ │ └── 3.jpg
│ └── ...
│
└── README.md # ไฟล์นี้
```
---

## 🔧 Requirements

- Python version: **>= 3.10, <= 3.12**

---

## 🚀 ฟังก์ชันหลัก

### ✅ สร้าง FAISS index (ครั้งแรกเท่านั้น)

1 : build_faiss()
2 : update_faiss()
3 : delete_vectors_by_name("Pun")
4 : Exiting Program
