Version reccomand >=3.10 , <=3.12

data/
-faiss_Store/
index.faiss #เก็บ vector
index.pkl   #เก็บ index vector ว่าเป็นใคร

#สร้างใหม่เท่านั้น
-person_img/
--name1/
    1.jpg
    2.jpg
    3.jpg
--name2/
    1.jpg
    2.jpg
    3.jpg
--name3/
    1.jpg
    2.jpg
    3.jpg

#add ชื่อเท่านั้น
-person_img_update/
--name1/
    1.jpg
    2.jpg
    3.jpg
--name2/
    1.jpg
    2.jpg
    3.jpg
--name3/
    1.jpg
    2.jpg
    3.jpg

Time
3 image / second

# สร้าง FAISS (ครั้งแรกเท่านั้น) 
# build_faiss()

# อัปเดต FAISS ด้วยรูปใหม่
# update_faiss()

# delete ใช่ชื่อ ที่จะลบออกไป
# delete_vectors_by_name("Pun")

