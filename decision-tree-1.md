# ต้นไม้การตัดสินใจ (Decision Tree)
ต้นไม้การตัดสินใจ (Decision Tree) เป็นอัลกอริทึมที่นิยมใน **Machine Learning** โดยเฉพาะการจัดประเภท (Classification) และการทำนายค่าเชิงตัวเลข (Regression) โดยโครงสร้างของต้นไม้จะเริ่มต้นจากโหนดราก (Root Node) และแตกแขนงออกเป็นโหนดย่อย (Branches) จนถึงโหนดใบ (Leaf Node)

##### ข้อดี: 
- เข้าใจง่าย, รองรับข้อมูลทั้งเชิงตัวเลขและเชิงหมวดหมู่
- สามารถแสดงเป็นแผนภาพเพื่อการนำเสนอได้ง่ายและดึงดูดสายตา ค่อนข้างเป็นไปตามสามัญสำนึก กรณีต้นไม้ไม่ซับซ้อน หรือมีความลึกมากเกินไป
- 
##### ข้อเสีย: 
- มีโอกาสเกิด Overfitting หากต้นไม้ซับซ้อนเกินไป


## การใช้งานต้นไม้การตัดสินใจใน Python ด้วย library sklearn

  

#### การจัดประเภท (Classification)

**โค้ดตัวอย่าง**

``` python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. โหลดชุดข้อมูลตัวอย่าง (Iris Dataset) จาก sklearn.datasets
data = load_iris()
X = data.data  # คุณลักษณะ (Features)
y = data.target  # เลเบล (Labels)

# 2. แบ่งข้อมูลออกเป็นชุดฝึกสอนและชุดทดสอบ ด้วย train_test_split ฟังก์ชันของ sklearn.model_selection 
# random_state กำหนดเป็นค่าหนึ่งๆ เพื่อสามารถทำซ้ำได้
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. สร้างโมเดลต้นไม้การตัดสินใจ
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# 4. ฝึกโมเดลด้วยข้อมูลฝึกสอน
model.fit(X_train, y_train)

# 5. ทำนายผลลัพธ์
y_pred = model.predict(X_test)

# 6. ประเมินผลลัพธ์
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 7. แสดงโครงสร้างต้นไม้
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plot_tree(model, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()
```

**คำอธิบายโค้ด**

1. **โหลดข้อมูลตัวอย่าง**

ใช้ชุดข้อมูล **Iris Dataset** ซึ่งเป็นข้อมูลการจัดประเภทของดอกไม้ 3 ชนิด (Class) ชุดข้อมูลนี้ได้รับความนิยมเป็นพิเศษเนื่องจากความเรียบง่าย เรียกได้ว่าเป็น Hello World dataset ของ Classification Algorithm เลยทีเดียว เรามีดอกไม้ทั้ง 3 ตามนี้
- Iris setosa
- Iris versicolor
- Iris virginica

แต่ละชนิดประกอบด้วย คุณลักษณะ 4 อย่าง (Features) ทั้งสี่นี้วัดเป็นเซนติเมตรทั้งหมด
- ความยาวกลีบเลี้ยง
- ความกว้างกลีบเลี้ยง
- ความยาวกลีบดอก
- ความกว้างกลีบดอก



2. **แบ่งข้อมูลออกเป็นชุดฝึกสอนและชุดทดสอบ**

แบ่งข้อมูลเพื่อใช้สำหรับฝึกโมเดล (70%) และทดสอบโมเดล (30%) และค่า random_state เพื่อให้การทำซ้ำได้ชุดแบ่งข้อมูลเหมือนเดิม จะได้ไม่กระทบต่อการปรับการฝึก Model เมื่อทำหลายครั้ง

3. **สร้างและฝึกโมเดลต้นไม้การตัดสินใจ**
ใช้ DecisionTreeClassifier โดยกำหนด:
- criterion='gini': ใช้ Gini Impurity เป็นตัววัดการแยกกิ่ง (split)
- max_depth=3: จำกัดความลึกของต้นไม้ไม่เกิน 3 ระดับ เพื่อลดความซับซ้อน

4. **ทำนายผลและประเมินความแม่นยำ**

ใช้ accuracy_score วัดความแม่นยำของโมเดล

5. **แสดงโครงสร้างต้นไม้**

ใช้ plot_tree เพื่อแสดงภาพต้นไม้ที่แสดงกฎการตัดสินใจในแต่ละโหนด

**ตัวอย่างผลลัพธ์**

• Accuracy: ประมาณ 95% ขึ้นอยู่กับข้อมูล

• กราฟต้นไม้การตัดสินใจจะแสดงกฎในแต่ละโหนด เช่น
