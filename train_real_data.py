from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import warnings
warnings.filterwarnings('ignore')

# 1. โหลดข้อมูลจริง (Wisesight Sentiment) จาก Hugging Face
print("ดาวน์โหลด Dataset Wisesight...")
dataset = load_dataset("wisesight_sentiment")

# 2. Data Preparation: เลือกเฉพาะ Positive (0) และ Negative (2) 
dataset = dataset.filter(lambda x: x['category'] in [0, 2])

# แปลง Label ให้ตรงกับโครงสร้างเดิม: Positive (0->1), Negative (2->0)
def format_data(example):
    return {"text": example["texts"], "label": 1 if example["category"] == 0 else 0}

print("กำลังจัดรูปแบบข้อมูล...")
dataset = dataset.map(format_data, remove_columns=["texts", "category"])

# สุ่มข้อมูลมาสอน 2,000 ประโยค และทดสอบ 400 ประโยค (เพื่อให้เทรนจบไว)
train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
test_dataset = dataset["validation"].shuffle(seed=42).select(range(400))

# 3. Tokenizer & Model
model_name = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("Tokenize ข้อมูล (หั่นคำ)...")
train_tokenized = train_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)

print("โหลด Pre-trained Model...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4. ตั้งค่าเทรน (Optimization สำหรับ VRAM 8GB โดยเฉพาะ)
training_args = TrainingArguments(
    output_dir="./results_real",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # ลดขนาด Batch ลงเพื่อกัน CUDA Out of Memory
    gradient_accumulation_steps=2,  # สะสม Gradient 2 รอบ (จำลองว่าใช้ Batch Size 8)
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    fp16=True,                      # ใช้ทศนิยม 16-bit ลดการกิน VRAM ลงครึ่งหนึ่ง
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
)

# 5. เริ่มเทรน
print("เริ่มกระบวนการ Fine-Tuning ด้วยข้อมูลจริง...")
trainer.train()

# 6. บันทึกโมเดล
trainer.save_model("./my_thai_model_real")
tokenizer.save_pretrained("./my_thai_model_real")
print("\nเสร็จสิ้นภารกิจ! บันทึกโมเดลที่ ./my_thai_model_real")