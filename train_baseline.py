import pandas as pd
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore') # ปิดแจ้งเตือนจุกจิกตอนรัน

# 1. โหลดข้อมูล (Data Loading)
print("Loading data...")
df = pd.read_csv('dataset.csv')

# 2. แบ่งข้อมูลสำหรับ "สอน" (Train) และ "ทดสอบ" (Test)
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# 3. เตรียมข้อมูล (Preprocessing & Feature Extraction)
# ใช้ PyThaiNLP ตัดคำ และ TF-IDF แปลงความถี่คำเป็นเวกเตอร์ตัวเลข
def thai_tokenizer(text):
    return word_tokenize(text, engine="newmm")

print("Vectorizing text...")
vectorizer = TfidfVectorizer(tokenizer=thai_tokenizer)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test) # สังเกตว่าใช้แค่ transform() ไม่ใช้ fit() กับ Test set

# 4. สร้างและสอนโมเดล (Model Training)
print("Training model...")
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 5. ประเมินผล (Evaluation)
print("\n--- Evaluation Results ---")
predictions = model.predict(X_test_vec)
print(classification_report(y_test, predictions))

# 6. ลองทดสอบกับข้อความใหม่
new_texts = ["ระบบใช้งานง่ายมาก ชอบสุดๆ", "หน้าจอกระตุก ทำงานต่อไม่ได้เลย"]
new_vec = vectorizer.transform(new_texts)
new_preds = model.predict(new_vec)
print(f"\nTest New Texts: {new_texts}")
print(f"Predictions (1=Good, 0=Bad): {new_preds}")