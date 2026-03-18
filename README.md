# 🇹🇭 Thai Text Classification with WangchanBERTa

โปรเจกต์ตัวอย่างสำหรับการสร้างและฝึกสอน (Fine-tuning) โมเดล AI เฉพาะทางภาษาไทย เพื่อแยกหมวดหมู่ข้อความ (Text Classification) โดยใช้สถาปัตยกรรม Transformer (WangchanBERTa) ออกแบบมาให้สามารถรันบนข้อจำกัด **VRAM 8GB (เช่น RTX 4060)** ได้จริงผ่านสภาพแวดล้อม WSL

> 📖 **คลังความรู้สำหรับผู้ปฏิบัติการ:** 
> * **[Learning Guide (`doc/learning_guide.md`)](./doc/learning_guide.md):** คู่มือเจาะลึกสถาปัตยกรรมโค้ดและการแก้ปัญหา Resource Optimization (VRAM 8GB)
> * **[Textbook Reference (`doc/textbook_reference.md`)](./doc/textbook_reference.md):** ตำราเรียนฉบับวิศวกรรม ผ่าตัดลึกถึงระดับไลบรารี PyTorch, Transfer Learning และกลไก Memory Bottlenecks แบบเจาะลึก

## 🛠️ โครงสร้างเทคโนโลยีและเครื่องมือ (Architecture & Tools)
* **Base Model:** `airesearch/wangchanberta-base-att-spm-uncased` (110M Parameters)
* **Framework:** PyTorch, Hugging Face `transformers` & `datasets`
* **Optimization:** Mixed Precision (FP16), Gradient Accumulation
* **Environment:** Ubuntu (WSL2), Python 3.12, CUDA 12.1

## 📂 โครงสร้างไฟล์เชิงสถาปัตยกรรม (Project Structure)
โปรเจกต์ถูกออกแบบตามหลักการ **Separation of Concerns (SoC)** แยกการทำงานที่ซับซ้อนออกจากกันดังนี้:

* `dataset.csv` - ไฟล์ข้อมูลจำลองขนาดเล็กมาก สำหรับใช้ทดสอบความสมบูรณ์ของไปป์ไลน์ระบบเบื้องต้น (Sanity Check)
* `train_baseline.py` - สคริปต์สร้างบรรทัดฐาน (Benchmark) ด้วยวิธี Classical ML (แบบตัดคำ PyThaiNLP + TF-IDF + Logistic Regression)
* `train_bert.py` - สคริปต์สเก๊ตช์โครงร่าง (Prototyping) สอน WangchanBERTa ด้วยชุดข้อมูลจำลอง
* `train_real_data.py` - แกนหลักของระบบ (Production Training) ดาวน์โหลดชุดข้อมูล Wisesight Sentiment ตัวเต็มมาสอนโมเดล พร้อมระบบรีดประสิทธิภาพ VRAM แบบเข้มข้น
* `predict_bert.py` - สคริปต์รันใช้งาน (Inference Engine) นำสมองที่เทรนเสร็จแล้วไปพยากรณ์ข้อมูล (ตัด AutoGrad ทิ้งเพื่อความเร็วสูงสุด)
* `doc/learning_guide.md` - เอกสารคู่มือทางเทคนิคสำหรับการศึกษาเชิงลึก
* `doc/textbook_reference.md` - ตำราเรียนฉบับวิศวกรรม ผ่าตัดโมเดลและฮาร์ดแวร์คอขวด
* `.gitignore` - ไฟล์ตั้งค่าของ Git สำหรับป้องกันโมเดลขนาดใหญ่หลุดขึ้น Repository

## 🚀 วิธีสับสวิตช์เริ่มทำงาน (Quick Start)

### 1. เตรียมสภาพแวดล้อมและติดตั้งส่วนเสริม (Setup Environment)
```bash
# 1. สร้างพื้นที่แยกส่วน (Virtual Environment)
python3 -m venv venv
source venv/bin/activate

# 2. ติดตั้งแกนประมวลผล PyTorch (รองรับแผงวงจร CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. โหลดคลังอาวุธสำหรับสร้าง AI และทำจัดเรียงข้อมูล
pip install transformers datasets scikit-learn pandas pythainlp sentencepiece protobuf accelerate
```

### 2. เรียกน้ำย่อยปลุกปั้นโมเดล AI (Training)
รันคำสั่งนี้เพื่อสตรีมข้อมูล Wisesight Sentiment นับพันข้อความมาไว้ในเครื่อง และเริ่มกระบวนการตีบวกโมเดล:

```bash
python train_real_data.py
```
> 💡 *หมายเหตุ: โมเดลที่เกิดจากการเรียนรู้นี้ จะประกอบร่างเป็นไฟล์เก็บไว้ในโฟลเดอร์ `./my_thai_model` และ `./my_thai_model_real`*

### 3. ทดสอบสั่งการโมเดล (Inference)
เมื่อโมเดลสอบผ่านและเทรนจบ ให้เช็กความฉลาดของมันด้วยประโยคทดสอบในสคริปต์นี้:

```bash
python predict_bert.py
```
