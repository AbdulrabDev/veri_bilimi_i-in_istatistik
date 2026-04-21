# 🎮 Video Game Sales Analysis & Statistical Hypothesis Testing

Bu proje, Sakarya Uygulamalı Bilimler Üniversitesi (SUBU) Bilişim Sistemleri Mühendisliği dersi kapsamında hazırlanmıştır. Veri seti üzerinde kapsamlı temizleme, istatistiksel analiz ve hipotez testleri uygulanmıştır.

## 📝 Proje Özeti (Project Overview)
Bu çalışma, dünya genelindeki video oyun satışlarını ve bu satışları etkileyen faktörleri (puanlar, türler, yıllar) analiz eder. Veri seti yaklaşık 16,000 oyun içermektedir.

---

## 🛠️ Yapılan İşlemler (Key Steps)

### 1. Veri Temizleme ve Ön İşleme (Data Cleaning)
* **Zaman Filtresi:** Sadece 2000 yılı ve sonrası modern oyunlara odaklanıldı.
* **Mükerrer Veri:** Tekrar eden satırlar silindi.
* **Veri Tipi Dönüştürme:** `User_Score` içindeki metinsel ifadeler (tbd) sayısal verilere çevrildi.
* **Eksik Veri Yönetimi (Imputation):** Kritik olmayan eksik puanlar, verinin aritmetik dengesini bozmamak için **Ortalama (Mean)** ile dolduruldu.
* **Örnekleme:** Analiz kalitesini artırmak için 1000 adet rastgele örneklem seçildi.

### 2. İstatistiksel Analiz (Statistical Analysis)
* **Normallik Testi:** `Shapiro-Wilk` testi kullanılarak verilerin normal dağılıp dağılmadığı kontrol edildi.
* **Varyans Homojenliği:** `Levene` testi uygulandı.
* **Görselleştirme:** Boxplot, Histogram ve Q-Q Plot grafikleri oluşturuldu.

### 3. Hipotez Testleri (Hypothesis Testing)
Veriler normal dağılmadığı için **Non-Parametric** testler tercih edilmiştir:
* **Mann-Whitney U Testi:** 2010 öncesi ve sonrası oyunların puan farkları analiz edildi.
* **Kruskal-Wallis Testi:** Oyun türlerinin kullanıcı puanları üzerindeki etkisi incelendi.

---

## 🚀 Kullanılan Teknolojiler (Tech Stack)
* **Python 3.x**
* **Pandas** (Veri manipülasyonu)
* **Seaborn & Matplotlib** (Görselleştirme)
* **Scipy & Statsmodels** (İstatistiksel testler)

---

## 📊 Örnek Çıktılar (Results)
Temizleme işlemi sonrası elde edilen final raporu:
- **Kalan Eksik Veri:** 0 (Tertemiz Veri)
- **Örneklem Sayısı:** 1000 Oyun

---

## 👤 Hazırlayan (Author)
* **ABDULRAB  ABDULRAB** - *Bilişim Sistemleri Mühendisliği Öğrencisi*
* Sakarya Üniversitesi (SAU)
