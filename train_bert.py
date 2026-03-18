import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import warnings
warnings.filterwarnings('ignore')

# 1. โหลดข้อมูลเดิม (ใช้ dataset.csv ไฟล์เดิมได้เลย)
df = pd.read_csv('dataset.csv')
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# 2. โหลด Tokenizer ของ WangchanBERTa
model_name = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. ฟังก์ชันแปลงข้อความเป็นตัวเลข (Tensor)
def tokenize_function(examples):
    # max_length=128 คือการจำกัดความยาวประโยค ช่วยประหยัด VRAM 
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("กำลัง Tokenize ข้อมูล...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. โหลดสถาปัตยกรรมโมเดลและตั้งค่า Output เป็น 2 หมวดหมู่ (0 และ 1)
print("กำลังโหลด Pre-trained Model...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 5. ตั้งค่าการเทรน (จุดนี้ออกแบบมาเพื่อจัดการ VRAM ของการ์ดจอโดยเฉพาะ)
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8, # หากตอนรันเจอ Error: CUDA Out of Memory ให้ลดเหลือ 4
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    fp16=True,  # หัวใจสำคัญ: เปิด Mixed Precision ลดการกิน VRAM บนการ์ดจอ
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# 6. เริ่มเทรน (ใช้ GPU)
print("เริ่มกระบวนการ Fine-Tuning ด้วย GPU...")
trainer.train()

# 7. บันทึกโมเดลไว้ใช้งานต่อ
trainer.save_model("./my_thai_model")
tokenizer.save_pretrained("./my_thai_model")
print("\nเทรนเสร็จสิ้น! บันทึกโมเดลเฉพาะทางของคุณไว้ที่โฟลเดอร์ ./my_thai_model แล้ว")