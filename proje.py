# ==========================================
#  Veri Temizleme ve İşleme
# ==========================================

# 1. قراءة البيانات
df = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')

# 2. تنظيف البيانات (Veri Temizleme)
# تحويل تقييم المستخدمين إلى أرقام ومعالجة النصوص الغريبة مثل 'tbd'
df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')

# حذف الأسطر التي تفتقد لبيانات أساسية (الاسم، السنة، النوع، المبيعات)
df.dropna(subset=['Name', 'Year_of_Release', 'Genre', 'Global_Sales'], inplace=True)

# تحويل السنة إلى رقم صحيح
df['Year_of_Release'] = df['Year_of_Release'].astype(int)

# 3. تعويض القيم المفقودة (Eksik Veri Doldurma)
# استبدال القيم الفارغة في تقييمات النقاد والجمهور بالمتوسط الحسابي
df['Critic_Score'] = df['Critic_Score'].fillna(df['Critic_Score'].mean())
df['User_Score'] = df['User_Score'].fillna(df['User_Score'].mean())

# 4. أخذ عينة عشوائية (Sampling / Örnekleme) 
# اختيار 1000 عينة لتمثيل المجتمع الإحصائي (أكثر من الـ 500 المطلوبة بالملف)
df_sample = df.sample(n=1000, random_state=42)

# ==========================================
# İstatistiksel Hesaplamalar ve Görselleştirmeler
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
stats_summary = df_sample[['Global_Sales', 'Critic_Score', 'User_Score']].describe()
print(stats_summary)

print(df_sample['Global_Sales'].skew())

correlation = df_sample[['Global_Sales', 'Critic_Score', 'User_Score']].corr()
print(correlation)

sns.histplot(df_sample['Global_Sales'], kde=True)
