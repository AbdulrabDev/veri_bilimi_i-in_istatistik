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

# ================================================
# Visualization (Boxplot - Countplot - Histogram)
# ================================================
# 1. Boxplot للمبيعات (كشف القيم الشاذة)
plt.figure(figsize=(6,4))
sns.boxplot(x=df_sample['Global_Sales'])
plt.title("Global Sales Boxplot")
plt.show()

# 2. Countplot للأنواع (Genre)
plt.figure(figsize=(8,5))
sns.countplot(x='Genre', data=df_sample)
plt.title("Game Genre Distribution")
plt.xticks(rotation=45)
plt.show()

# ==========================================
 YÜKSEK SATIŞLI OYUN ORANI
# ==========================================
mean_sales = df_sample['Global_Sales'].mean()
percentage = (len(df_sample[df_sample['Global_Sales'] > mean_sales]) / len(df_sample)) * 100
print(f"\n--- [BÖLÜM 4]: EK ANALİZ ---")
print(f"Ortalama üstü satış yapan oyunların oranı: %{percentage:.2f}")



# ==========================================
# 3. VARSAYIM TESTLERİ (ASSUMPTION TESTS)
# ==========================================
print("\n--- [BÖLÜM 2]: VARSAYIM TESTLERİ ---")

# Shapiro-Wilk Normallik Testi
shapiro_stat, shapiro_p = stats.shapiro(df_sample['Global_Sales'])
print(f"1. Shapiro-Wilk Testi P-Değeri: {shapiro_p:.20f}")
print("NEDEN: Verilerin normal dağılıp dağılmadığını belirlemek, doğru test yöntemini seçmek için zorunludur.")
print(f"YORUM: P < 0.05 olduğu için veriler normal dağılmamaktadır.\n")

# Levene Testi (Varyans Homojenliği)
action_sales = df[df['Genre'] == 'Action']['Global_Sales']
sports_sales = df[df['Genre'] == 'Sports']['Global_Sales']
_, levene_p = stats.levene(action_sales.sample(100), sports_sales.sample(100))
print(f"2. Levene Testi P-Değeri: {levene_p:.10f}")
print("NEDEN: Grupların varyanslarının eşitliğini kontrol etmek, karşılaştırma testlerinin güvenilirliği için gereklidir.")
print("-" * 50)

# ==========================================
# 4. HİPOTEZ TESTLERİ (INFERENTIAL STATISTICS)
# ==========================================
print("\n--- [BÖLÜM 3]: HİPOTEZ TESTLERİ ---")

# Spearman Korelasyon Analizi
corr_val, spearman_p = stats.spearmanr(df_sample['Critic_Score'], df_sample['Global_Sales'])
print(f"1. Spearman Korelasyonu: {corr_val:.3f} | P-Değeri: {spearman_p:.10f}")

plt.figure()
sns.regplot(x='Critic_Score', y='Global_Sales', data=df_sample, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title("Eleştirmen Puanı ve Küresel Satış İlişkisi")
plt.show()

print("RAPOR: Pozitif korelasyon, eleştirmen puanları arttıkça satışların artma eğiliminde olduğunu kanıtlar.")

# Bağımsız Örneklem T-Testi
t_stat, t_p = stats.ttest_ind(action_sales.sample(100), sports_sales.sample(100))
print(f"\n2. Bağımsız T-Testi (Action vs Sports) P-Değeri: {t_p:.4f}")

plt.figure()
sns.boxplot(x='Genre', y='Global_Sales', data=df[df['Genre'].isin(['Action', 'Sports'])])
plt.ylim(0, 5)
plt.title("Aksiyon ve Spor Oyunları Satış Karşılaştırması")
plt.show()

def karar_ver(p):
    return "İstatistiksel olarak anlamlı bir fark vardır (H0 reddedildi)." if p < 0.05 else "Anlamlı bir fark yoktur (H0 reddedilemedi)."

print(f"KARAR: {karar_ver(t_p)}")

