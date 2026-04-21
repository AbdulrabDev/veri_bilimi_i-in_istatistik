import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, spearmanr
import statsmodels.api as sm
import numpy as np

# =============================================================================
#        BÖLÜM 1 : VERİ ANALİZİ, ÖN İŞLEME VE TEMİZLEME 
# =============================================================================

# Bu bölümün amacı, ham veriyi analiz edilebilir, temiz ve güvenilir bir 
# hale getirmektir. Eksik veriler yönetilmiş ve mantıksal filtreleme yapılmıştır.


# 1. ADIM: Veri Setinin Sisteme Yüklenmesi
# Video oyunları satış verilerini içeren CSV dosyası pandas kütüphanesi ile okunur.
df = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')

# 3. ADIM: İlk Eksik Veri Analizi (Eksik Veri Oranlarının Tespiti)
# Hangi sütunlarda ne kadar veri kaybı olduğunu anlamak için yüzde hesabı yapılır.
# Bu adım, temizleme stratejimizi belirlemek için kritik öneme sahiptir.
missing_ratio = (df.isnull().sum() / len(df)) * 100
print("--- Sütun Bazlı Eksik Veri Oranları (%) ---")
print(missing_ratio.round(2))
print("-" * 50)

# 2. ADIM: Kritik Olmayan Satırların Hesaplanması ve Kaldırılması
# İsim, Yıl, Tür ve Global Satış gibi temel bilgilerde eksiklik olan satırlar önce hesaplanır sonra silinir.
# Çünkü bu bilgiler olmadan analiz yapmak mümkün değildir.

missing_critical = df[['Name', 'Year_of_Release', 'Genre', 'Global_Sales']].isnull().any(axis=1).sum()
print(f"--- Temel verisi eksik olan satır sayısı: {missing_critical} (Silinecek) ---")

# Silme
df.dropna(subset=['Name', 'Year_of_Release', 'Genre', 'Global_Sales'], inplace=True)
print("--- Temel verisi eksik olan satırlar başarıyla silindi ---")


# 4. ADIM: Mükerrer Veri Kontrolü (Duplicate Rows)
# Veri setinde tamamen aynı olan satırlar varsa bunlar tespit edilir ve silinir.
# Bu işlem, istatistiksel sonuçların yapay olarak sapmasını önler.
duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    df.drop_duplicates(inplace=True)
    print(f"--- {duplicate_count} adet mükerrer satır başarıyla temizlendi ---")
    

# 5. ADIM: Mantıksal Filtreleme (Yıl Bazlı Filtreleme)
# Analizin güncel kalması ve modern oyun pazarını yansıtması için 
# sadece 2000 yılı ve sonrasında çıkan oyunlar veri setine dahil edilir.
df = df[df['Year_of_Release'] >= 2000]
print(f"--- Filtreleme Sonrası (2000+) Toplam Oyun Sayısı: {len(df)} ---")


# 5. ADIM: Veri Tiplerinin Düzenlenmesi ve Metin Temizliği
# User_Score sütunu normalde 'object' (metin) tipindedir çünkü içinde 'tbd' yazıları vardır.
# to_numeric fonksiyonu ile 'tbd' ifadeleri NaN (boş veri) yapılır ve sütun sayısal olur.
df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')


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
print("--- FİNAL TEMİZLİK KONTROL RAPORU ---")
final_check = df_sample[['Critic_Score', 'User_Score']].isnull().sum()
print("Kalan Eksik Veri Sayısı (0 olmalı):")
print(final_check)
print("="*50)

print(f"\n--- Analiz için toplam {len(df_sample)} adet tertemiz veri hazırlandı. ---")

# =============================================================================
#         BÖLÜM 2 : İstatistiksel Hesaplamalar ve Görselleştirmeler
# =============================================================================

# Bu bölümde veri setinin betimsel istatistik analizi yapılmaktadır.
# Amaç; satışlar ve puanlar gibi temel değişkenlerin genel yapısını,
# dağılım özelliklerini ve değişkenler arasındaki ilişkileri incelemektir.

# ---------------------------------------------------------------------
# 1. ADIM: Analizde kullanılacak sayısal sütunların seçilmesi
# Global_Sales analizimizin temel hedef değişkenidir.
# Critic_Score ve User_Score ise satışlarla ilişkisi incelenecek puan değişkenleridir.
selected_cols = ['Global_Sales', 'Critic_Score', 'User_Score']

# Sadece gerekli sütunları alarak daha okunabilir bir alt veri çerçevesi oluşturuyoruz.
analysis_df = df_sample[selected_cols]

# ---------------------------------------------------------------------
# 2. ADIM: Temel betimsel istatistiklerin hesaplanması
# describe() fonksiyonu; gözlem sayısı, ortalama, standart sapma,
# minimum, maksimum ve çeyreklik değerleri verir.
print("\n--- TEMEL BETİMSEL İSTATİSTİKLER ---")
stats_summary = analysis_df.describe()
print(stats_summary)

# ---------------------------------------------------------------------
# 3. ADIM: Varyans hesaplanması
# Varyans, verinin ortalama etrafında ne kadar yayıldığını gösterir.
print("\n--- VARYANS DEĞERLERİ ---")
variance_values = analysis_df.var()
print(variance_values)

# ---------------------------------------------------------------------
# 4. ADIM: Çarpıklık (Skewness) analizi
# Çarpıklık değeri, dağılımın simetrik olup olmadığını gösterir.
# Pozitif değer sağa çarpıklığı, negatif değer sola çarpıklığı ifade eder.
print("\n--- ÇARPIKLIK (SKEWNESS) ANALİZİ ---")
skew_values = analysis_df.skew()
print(skew_values)

# Global_Sales için özel yorum
global_sales_skew = analysis_df['Global_Sales'].skew()
if global_sales_skew > 1:
    print("YORUM: Global_Sales değişkeni belirgin şekilde sağa çarpıktır.")
elif global_sales_skew > 0:
    print("YORUM: Global_Sales değişkeni hafif sağa çarpıktır.")
elif global_sales_skew < 0:
    print("YORUM: Global_Sales değişkeni sola çarpıktır.")
else:
    print("YORUM: Global_Sales dağılımı yaklaşık simetriktir.")

# ---------------------------------------------------------------------
# 5. ADIM: Korelasyon matrisinin oluşturulması
# Bu analiz, satışlar ile eleştirmen ve kullanıcı puanları arasındaki
# doğrusal ilişkinin gücünü ve yönünü gösterir.
print("\n--- KORELASYON MATRİSİ ---")
correlation = analysis_df.corr()
print(correlation)

# Critic_Score ile Global_Sales arasındaki ilişkiye kısa yorum
critic_sales_corr = correlation.loc['Global_Sales', 'Critic_Score']
user_sales_corr = correlation.loc['Global_Sales', 'User_Score']

print("\n--- KORELASYON YORUMU ---")
print(f"Critic_Score ile Global_Sales korelasyonu: {critic_sales_corr:.3f}")
print(f"User_Score ile Global_Sales korelasyonu  : {user_sales_corr:.3f}")

# ---------------------------------------------------------------------
# 6. ADIM: Global_Sales histogram grafiği
# Histogram, verinin hangi aralıklarda yoğunlaştığını gösterir.
# KDE eğrisi ise dağılımın genel şeklini daha yumuşak biçimde görmemizi sağlar.
plt.figure(figsize=(8, 5))
sns.histplot(analysis_df['Global_Sales'], kde=True, bins=30, color='steelblue')
plt.title("Global Sales Dağılımı")
plt.xlabel("Global Sales")
plt.ylabel("Frekans")
plt.grid(True, alpha=0.3)
plt.show()

# ---------------------------------------------------------------------
# 7. ADIM: Critic_Score histogram grafiği
# Eleştirmen puanlarının örneklem içinde nasıl dağıldığını görselleştiriyoruz.
plt.figure(figsize=(8, 5))
sns.histplot(analysis_df['Critic_Score'], kde=True, bins=25, color='darkgreen')
plt.title("Critic Score Dağılımı")
plt.xlabel("Critic Score")
plt.ylabel("Frekans")
plt.grid(True, alpha=0.3)
plt.show()

# ---------------------------------------------------------------------
# 8. ADIM: User_Score histogram grafiği
# Kullanıcı puanlarının yoğunlaştığı aralıkları görmek için çizilir.
plt.figure(figsize=(8, 5))
sns.histplot(analysis_df['User_Score'], kde=True, bins=25, color='darkorange')
plt.title("User Score Dağılımı")
plt.xlabel("User Score")
plt.ylabel("Frekans")
plt.grid(True, alpha=0.3)
plt.show()

# ---------------------------------------------------------------------
# 9. ADIM: Global_Sales boxplot grafiği
# Aykırı değerlerin varlığını kontrol etmek için kullanılır.
plt.figure(figsize=(8, 4))
sns.boxplot(x=analysis_df['Global_Sales'], color='lightcoral')
plt.title("Global Sales Boxplot")
plt.xlabel("Global Sales")
plt.grid(True, alpha=0.3)
plt.show()

# ---------------------------------------------------------------------
# 10. ADIM: Türlere göre oyun sayısı (Countplot)
# Hangi oyun türünün örneklemde daha fazla temsil edildiğini gösterir.
plt.figure(figsize=(10, 5))
sns.countplot(x='Genre', data=df_sample, order=df_sample['Genre'].value_counts().index, palette='viridis')
plt.title("Oyun Türlerinin Dağılımı")
plt.xlabel("Genre")
plt.ylabel("Oyun Sayısı")
plt.xticks(rotation=45)
plt.grid(True, axis='y', alpha=0.3)
plt.show()

# ---------------------------------------------------------------------
# 11. ADIM: Korelasyon ısı haritası (Heatmap)
# Korelasyon matrisini görsel olarak yorumlamayı kolaylaştırır.
plt.figure(figsize=(6, 4))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Korelasyon Isı Haritası")
plt.show()


# =============================================================================
#            BÖLÜM 3 : VARSAYIM TESTLERİ (ASSUMPTION TESTS)
# =============================================================================
print("\n--- [BÖLÜM 3]: VARSAYIM TESTLERİ ---")

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

scores = ['User_Score', 'Critic_Score']

print("==== PUANLAR İÇİN NORMALLİK testi (SHAPIRO-WILK) ====\n")

for score in scores:
    print(f"--- {score} Analizi ---")
    
    # Shapiro-WILK Testi
    shapiro_stat, shapiro_p = stats.shapiro(df_sample[score])
    print(f"P-Değeri: {shapiro_p:.10f}")
    
    #  (Histogram & Q-Q Plot)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram & KDE
    sns.histplot(df_sample[score], kde=True, color='purple', ax=ax1)
    ax1.set_title(f'{score} Dağılımı (Histogram)')
    
    # Q-Q Plot
    stats.probplot(df_sample[score], dist="norm", plot=ax2)
    ax2.set_title(f'{score} için Q-Q Plot')
    
    plt.show()
    
    # Sonuç
    if shapiro_p < 0.05:
        print(f"YORUM: P-değeri ({shapiro_p:.4f}) < 0.05 olduğu için {score} normal dağılım göstermemektedir.")
        print(f"Q-Q Plot üzerindeki sapmalar, verilerin normal dağılım çizgisinden uzaklaştığını kanıtlar.\n")
    else:
        print(f"YORUM: P-değeri ({shapiro_p:.4f}) > 0.05 olduğu için {score} normal dağılıma uygundur.\n")
    
    print("-" * 60)
print("==== Years_of release İÇİN KAPSAMLI NORMALLİK ANALİZİ ====\n")

target_variables =['Year_of_Release','NA_Sales','EU_Sales']
for var in target_variables:
    print(f"--- ANALİZ EDİLEN DEĞİŞKEN: {var} ---")
    
    # [أ] Shapiro-Wilk Testi (İstatistiksel Kontrol)
    stat, p_val = stats.shapiro(df_sample[var])
    print(f"Shapiro-Wilk P-Değeri: {p_val:.10f}")
    
    #  Görselleştirme (Histogram ve Q-Q Plot yan yana)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # Histogram & Dağılım Eğrisi
    sns.histplot(df_sample[var], kde=True, color='darkred' if var == 'Year_of_Release' else 'darkblue', ax=ax1)
    ax1.set_title(f'{var} Dağılımı (Histogram)')
    
    # Q-Q Plot
    stats.probplot(df_sample[var], dist="norm", plot=ax2)
    ax2.set_title(f'{var} için Q-Q Plot')
    
    plt.tight_layout()
    plt.show()
    
    # Yorumlama
    print(f"NEDEN KULLANILDI: {var} değişkeninin normal dağılım sergileyip sergilemediğini görsel ve numerik olarak doğrulamak için.")
    
    if p_val < 0.05:
        print(f"SONUÇ: P-değeri < 0.05 olduğu için {var} normal dağılım göstermemektedir.")
        print(f"YORUM: Grafikler üzerindeki sapmalar, verinin parametrik test koşullarını tam olarak sağlamadığını kanıtlar.\n")
    else:
        print(f"SONUÇ: P-değeri > 0.05 olduğu için {var} normal dağılıma uygundur.\n")
    
    print("-" * 80 + "\n")
# Levene Testi (Varyans Homojenliği)
action_sales = df_sample[df_sample['Genre'] == 'Action']['Global_Sales']
sports_sales = df_sample[df_sample['Genre'] == 'Sports']['Global_Sales']
_, levene_p = stats.levene(action_sales.sample(100), sports_sales.sample(100))
print(f"2. Levene Testi P-Değeri: {levene_p:.10f}")
print("NEDEN: Grupların varyanslarının eşitliğini kontrol etmek, karşılaştırma testlerinin güvenilirliği için gereklidir.")
print("-" * 50)

# =============================================================================
#                    BÖLÜM 4 : HİPOTEZ TESTLERİ
# =============================================================================

alpha = 0.05 # Anlamlılık düzeyi %5

print("\n" + "="*50)
print("--- HİPOTEZ TESTLERİ ---")
print("="*50)

# ==========================================
# HİPOTEZ 1: MANN-WHITNEY U TESTİ
# ==========================================

print("\n--- HİPOTEZ 1: Mann-Whitney U Testi ---")
print("Soru: 2010 öncesi oyunların Critic_Score'u daha yüksek mi?")

# NEDEN MANN-WHITNEY U?
# 1. Veriler normal dağılmıyor (Shapiro p < 0.05)
# 2. Sadece 2 grubumuz var (2010 öncesi / sonrası)
# 3. Normal dağılım gerektirmeyen 2 grup testi

# Grupları oluştur
group_before = df_sample[df_sample['Year_of_Release'] < 2010]['Critic_Score'].dropna()
group_after  = df_sample[df_sample['Year_of_Release'] >= 2010]['Critic_Score'].dropna()

# Grup bilgilerini yazdır
print(f"\n2010 Öncesi Oyun Sayısı   : {len(group_before)}")
print(f"2010 Sonrası Oyun Sayısı  : {len(group_after)}")
print(f"2010 Öncesi Ortalama Puan : {group_before.mean():.2f}")
print(f"2010 Sonrası Ortalama Puan: {group_after.mean():.2f}")

# Testi uygula
stat, p_val = mannwhitneyu(group_before, group_after, alternative='greater')

# Sonucu 
print(f"""
{'='*50}
HİPOTEZ 1 DETAYLI SONUÇ
{'='*50}
Hipotez   : 2010 Öncesi ve Sonrası Critic_Score
Test      : Mann-Whitney U
İstatistik: {stat:.2f} (658 × 342 = 224,936 toplam karşılaşma yapıldı)
p-değeri  : {p_val:.8f}
Karar     : {"H0 Reddedilir ✅" if p_val < alpha else "H0 Reddedilemez ❌"}
{'='*50}
""")
# =====================
# HİPOTEZ 1 YORUM
# =====================
#Sonuç olarak, analizimizde anlamlı bir fark bulunamamıştır
#ancak bu durum iki grup arasında kesinlikle fark yoktur anlamına gelmez
#Yani fark olabilir, ama biz bunu istatistiksel olarak kanıtlayamadık


# ==========================================
# HİPOTEZ 2: KRUSKAL-WALLIS TESTİ
# ==========================================

# NEDEN KRUSKAL-WALLIS?
# 1. Veriler normal dağılmıyor (Shapiro p < 0.05)
# 2. 12 farklı türümüz var
# 3. Normal dağılım gerektirmeyen çok grup testi
# ANOVA normal dağılım varsaydığı için kullanamayız.
# Mann-Whitney U sadece 2 grup için kullanılır.

print("\n--- HİPOTEZ 2: Kruskal-Wallis Testi ---")
print("Soru: Oyun türü kullanıcı puanlarını etkiliyor mu?")

# Role-Playing → RPG olarak yeniden adlandır
df_sample['Genre'] = df_sample['Genre'].replace({'Role-Playing': 'RPG'})

# Tüm türleri al
all_genres = sorted(df_sample['Genre'].unique())

# Her tür için grup oluştur ve bilgileri yazdır
groups_genre = []
print("Tür Bazlı Ortalama Kullanıcı Puanları:")
print("-" * 40)

for genre in all_genres:
    group = df_sample[df_sample['Genre'] == genre]['User_Score'].dropna()
    groups_genre.append(group)
    print(f"{genre:<15} → Oyun Sayısı: {len(group):>3}, Ortalama: {group.mean():.2f}")

# En yüksek ve en düşük türü bul
genre_means = df_sample.groupby('Genre')['User_Score'].mean()
best_genre  = genre_means.idxmax()
worst_genre = genre_means.idxmin()
print(f"\nEn Yüksek Puanlı Tür: {best_genre} ({genre_means[best_genre]:.2f})")
print(f"En Düşük Puanlı Tür : {worst_genre} ({genre_means[worst_genre]:.2f})")

# Testi uygula
stat, p_val = kruskal(*groups_genre)

# Sonucu 
print(f"""
{'='*50}
HİPOTEZ 2 DETAYLI SONUÇ
{'='*50}
Hipotez   : Oyun Türünün User_Score Üzerindeki Etkisi
Test      : Kruskal-Wallis
İstatistik: {stat:.2f}
p-değeri  : {p_val:.8f}
Karar     : {"H0 Reddedilir ✅" if p_val < alpha else "H0 Reddedilemez ❌"}
{'='*50}
""")

# =====================
# HİPOTEZ 2 YORUM
# =====================
# Oyun türünün kullanıcı deneyim puanları (User_Score)
# üzerinde istatistiksel olarak anlamlı bir etkisi bulunamamıştır.

# ==========================================
# HİPOTEZ 3: SPEARMAN KORELASYON TESTİ
# ==========================================

# NEDEN SPEARMAN?
# 1. Veriler normal dağılmıyor (Shapiro p < 0.05)
# 2. İki sayısal değişken arasındaki ilişkiyi ölçüyoruz
# 3. Mann-Whitney ve Kruskal-Wallis grup karşılaştırması
# için kullanılır, ilişki ölçümü için uygun değildir.


print("\n--- HİPOTEZ 3: Spearman Korelasyon Testi ---")
print("Soru: Eleştirmen puanı küresel satışları etkiliyor mu?")


# Spearman Korelasyon Testini uygula
corr, p_val = spearmanr(df_sample['Critic_Score'], df_sample['Global_Sales'])

# Korelasyon gücünü belirle
if abs(corr) >= 0.7:
    strength = "Güçlü"
elif abs(corr) >= 0.4:
    strength = "Orta"
elif abs(corr) >= 0.1:
    strength = "Zayıf"
else:
    strength = "Çok Zayıf"

direction = "Pozitif" if corr > 0 else "Negatif"

# Sonucu 
print(f"""
{'='*50}
HİPOTEZ 3 DETAYLI SONUÇ
{'='*50}
Hipotez         : Critic_Score ve Global_Sales İlişkisi
Test            : Spearman Korelasyon
Korelasyon (r)  : {corr:.4f}
İlişki Gücü     : {strength} {direction}
p-değeri        : {p_val:.8f}
Karar           : {"H0 Reddedilir ✅" if p_val < alpha else "H0 Reddedilemez ❌"}
{'='*50}
""")

# ===================
# HİPOTEZ 3 YORUM
# ===================
# Eleştirmen puanı (Critic_Score) ile küresel satışlar
# (Global_Sales) arasında istatistiksel olarak anlamlı
# pozitif bir ilişki bulunmuştur.
# Bu sonuç, yüksek eleştirmen puanı alan oyunların
# daha fazla satış yapma eğiliminde olduğunu gösteriyor.
 
