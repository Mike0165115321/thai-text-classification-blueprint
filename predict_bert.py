import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. โหลดโมเดลและ Tokenizer ที่เราเทรนและเซฟไว้
model_path = "./my_thai_model_real"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 2. ข้อความใหม่ที่ต้องการทดสอบ
new_texts = ["ระบบใช้งานง่ายมาก ชอบสุดๆ", "หน้าจอกระตุก ทำงานต่อไม่ได้เลย"]

# 3. แปลงข้อความเป็น Tensor ตามมาตรฐานของ WangchanBERTa
inputs = tokenizer(new_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

# 4. พยากรณ์ผลลัพธ์
model.eval() # สลับเป็นโหมดใช้งานจริง (ปิดระบบสุ่ม Dropout/BatchNorm)
with torch.no_grad(): # ปิดการคำนวณ Gradient เพื่อประหยัด Memory และรันเร็วขึ้น
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

print(f"ข้อความ: {new_texts}")
print(f"ผลลัพธ์ (1=ดี, 0=แย่): {predictions.tolist()}")