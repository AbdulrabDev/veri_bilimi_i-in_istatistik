import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
#         VERİ ANALİZİ, ÖN İŞLEME VE TEMİZLEME (SORUMLU: 1. KİŞİ)
# =============================================================================

# Bu bölümün amacı, ham veriyi analiz edilebilir, temiz ve güvenilir bir 
# hale getirmektir. Eksik veriler yönetilmiş ve mantıksal filtreleme yapılmıştır.


# 1. ADIM: Veri Setinin Sisteme Yüklenmesi
# Video oyunları satış verilerini içeren CSV dosyası pandas kütüphanesi ile okunur.
df = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')

# 2. ADIM: İlk Eksik Veri Analizi (Eksik Veri Oranlarının Tespiti)
# Hangi sütunlarda ne kadar veri kaybı olduğunu anlamak için yüzde hesabı yapılır.
# Bu adım, temizleme stratejimizi belirlemek için kritik öneme sahiptir.
missing_ratio = (df.isnull().sum() / len(df)) * 100
print("--- 1. KİŞİ: Sütun Bazlı Eksik Veri Oranları (%) ---")
print(missing_ratio.round(2))
print("-" * 50)

# 3. ADIM: Mantıksal Filtreleme (Yıl Bazlı Filtreleme)
# Analizin güncel kalması ve modern oyun pazarını yansıtması için 
# sadece 2000 yılı ve sonrasında çıkan oyunlar veri setine dahil edilir.
df = df[df['Year_of_Release'] >= 2000]
print(f"--- Filtreleme Sonrası (2000+) Toplam Oyun Sayısı: {len(df)} ---")

# 4. ADIM: Mükerrer Veri Kontrolü (Duplicate Rows)
# Veri setinde tamamen aynı olan satırlar varsa bunlar tespit edilir ve silinir.
# Bu işlem, istatistiksel sonuçların yapay olarak sapmasını önler.
duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    df.drop_duplicates(inplace=True)
    print(f"--- {duplicate_count} adet mükerrer satır başarıyla temizlendi ---")

# 5. ADIM: Veri Tiplerinin Düzenlenmesi ve Metin Temizliği
# User_Score sütunu normalde 'object' (metin) tipindedir çünkü içinde 'tbd' yazıları vardır.
# to_numeric fonksiyonu ile 'tbd' ifadeleri NaN (boş veri) yapılır ve sütun sayısal olur.
df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')

# 6. ADIM: Kritik Olmayan Satırların Kaldırılması
# İsim, Yıl, Tür ve Global Satış gibi temel bilgilerde eksiklik olan satırlar silinir.
# Çünkü bu bilgiler olmadan analiz yapmak mümkün değildir.
df.dropna(subset=['Name', 'Year_of_Release', 'Genre', 'Global_Sales'], inplace=True)

# 7. ADIM: Eksik Verilerin Ortalama (Mean) ile Doldurulması (Imputation)
# Critic_Score ve User_Score sütunlarında çok fazla eksik veri olduğu için 
# bu satırları silmek yerine verinin aritmetik dengesini bozmadan ortalama ile dolduruyoruz.
df['Critic_Score'] = df['Critic_Score'].fillna(df['Critic_Score'].mean())
df['User_Score'] = df['User_Score'].fillna(df['User_Score'].mean())

# 8. ADIM: Örnekleme (Sampling)
# Çok büyük veri setlerinde çalışmak karmaşıklığı artırabilir. 
# Bu nedenle toplumun özelliklerini yansıtan 1000 adet rastgele örneklem seçilir.
df_sample = df.sample(n=1000, random_state=42)

# 9. ADIM: SON KONTROL VE TEMİZLİK TESTİ (Nispeten Önemli Adım)
# Bu adımda, yaptığımız tüm temizlik işlemlerinin başarılı olup olmadığı test edilir.
# Puan sütunlarında 0 (sıfır) sonucu çıkması, verinin tamamen temizlendiğini kanıtlar.
print("\n" + "="*50)
print("--- 1. KİŞİ: FİNAL TEMİZLİK KONTROL RAPORU ---")
final_check = df_sample[['Critic_Score', 'User_Score']].isnull().sum()
print("Kalan Eksik Veri Sayısı (0 olmalı):")
print(final_check)
print("="*50)

print(f"\n--- Analiz için toplam {len(df_sample)} adet tertemiz veri hazırlandı. ---")

# ==========================================
# İstatistiksel Hesaplamalar ve Görselleştirmeler
# ==========================================

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

# B. Q-Q Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
stats.probplot(df_sample['Global_Sales'], dist="norm", plot=plt)
plt.title('Q-Q Plot (Normallik Kontrolü)')

# C. Histogram
plt.subplot(1, 2, 2)
sns.histplot(df_sample['Global_Sales'], kde=True, color='teal')
plt.title('Satış Dağılımı (Histogram)')
plt.show()

print("NEDEN KULLANILDI: Verilerin normal dağılıp dağılmadığını hem numerik (Shapiro) hem de görsel (Q-Q Plot) olarak doğrulamak için kullanıldı.")
print("RAPOR: P < 0.05 ve Q-Q Plot'taki sapmalar, verilerin normal dağılmadığını ve sağa çarpık olduğunu kanıtlar.\n")

# Levene Testi (Varyans Homojenliği)
action_sales = df_sample[df_sample['Genre'] == 'Action']['Global_Sales']
sports_sales = df_sample[df_sample['Genre'] == 'Sports']['Global_Sales']
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

