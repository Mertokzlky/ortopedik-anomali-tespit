# 🏥 Ortopedik Anomali Tespit Sistemi

Makine öğrenmesi destekli bir **karar destek sistemi (KDS)**: hastaların radyolojik ölçümlerinden (pelvik açılar, lomber lordoz vb.) yola çıkarak omurga yapısını **Normal** ya da **Anormal** (bel fıtığı / spondilolistezis riski) olarak sınıflandırır. Arayüz [Streamlit](https://streamlit.io) ile geliştirilmiştir.

> *English:* A machine-learning decision-support system that classifies spinal condition (Normal / Abnormal) from biomechanical measurements, built with scikit-learn and an interactive Streamlit UI.

🔗 **Canlı demo:** https://ortopedik-anomali-tespit.streamlit.app/

![Tahmin ekranı](docs/screenshots/tahmin-sistemi.png)

## Özellikler

- 🩺 **Tahmin sistemi:** 6 radyolojik ölçümü kaydırma çubuklarıyla girip anlık tahmin alma
- 📊 **Güven oranı grafiği:** modelin "Normal" / "Anormal" olasılıklarını görselleştirir
- 📈 **Veri analizi sekmesi:** veri setine genel bakış, istatistikler, hasta dağılımı, değişkenler arası saçılım grafiği
- 🔥 **Korelasyon matrisi:** özellikler arasındaki ilişkiyi ısı haritasıyla gösterir
- 🧩 **Karmaşıklık matrisi (confusion matrix):** modelin doğru/yanlış tahmin dağılımı
- ⚙️ **3 algoritma karşılaştırması:** Random Forest, SVM ve KNN eğitilip test başarıları karşılaştırılır
- 🔀 **2 sınıf / 3 sınıf modu:** kullanıcı, Normal/Anormal (2 sınıf) veya Normal/Disk Hernisi/Spondilolistezis (3 sınıf) modelleri arasında geçiş yapabilir; tüm sekmeler (tahmin, analiz, toplu tahmin) seçilen moda göre çalışır
- 📁 **Toplu tahmin:** birden fazla hastanın verisini içeren bir CSV dosyası yükleyip hepsi için aynı anda tahmin alma, sonuçları CSV olarak indirme
- ⚠️ **Uç değer uyarısı:** girilen bir ölçüm, hastaların %90'ının bulunduğu tipik aralığın dışındaysa kullanıcı uyarılır

| Tahmin sonucu | Korelasyon matrisi |
| --- | --- |
| ![Tahmin sonucu](docs/screenshots/tahmin-sonucu.png) | ![Korelasyon matrisi](docs/screenshots/korelasyon-matrisi.png) |

## Veri seti ve kaynak

- **Veri seti:** [Vertebral Column Dataset](https://www.kaggle.com/datasets/caesarlupum/vertebralcolumndataset) (Kaggle üzerinden, orijinal kaynak: UCI Machine Learning Repository / Dr. Henrique da Mota)
- **Kayıt sayısı:** 310 satır, 6 özellik + 1 hedef değişken (`column_2C.csv` — ikili sınıflandırma: Normal / Abnormal)
- **Literatür karşılaştırması:** Reshi, A. A. et al. (2021), *"Diagnosis of vertebral column pathologies using concatenated resampling with machine learning algorithms"*, PeerJ Computer Science. ([makaleye bağlantı](https://peerj.com/articles/cs-547/)) — bu çalışmada Random Forest %88, SVM %85 doğruluk elde etmiştir; bu projede KNN ile elde edilen %83.87 sonucu literatürle tutarlıdır.

## Model performansı

| Algoritma | Test Doğruluğu (tek ayrım) | 5-Katlı Çapraz Doğrulama |
| --- | --- | --- |
| **KNN (En Yakın Komşu)** 🏆 | **%83.87** | %84.19 (± %2.96) |
| SVM (Destek Vektör) | %80.65 | %85.48 (± %3.68) |
| Random Forest | %77.42 | %84.19 (± %3.13) |

`model_egit.py` çalıştırıldığında veriler %80/%20 (eğitim/test) olarak bölünür (`random_state=42`, tekrarlanabilir sonuçlar için), üç algoritma da eğitilir ve en başarılı model (`fiziktedavi_model.pkl`) ile tüm skorlar (`model_skorlari.pkl`) diske kaydedilir.

Tek bir train/test ayrımı şansa bağlı sonuç verebileceğinden, aynı zamanda **5 katlı çapraz doğrulama** (`StratifiedKFold` + `cross_val_score`) da uygulanır: veri 5 farklı şekilde bölünüp her model 5 kez test edilir. Sonuçlar tek ayrımla tutarlı çıkmıştır; SVM'in çapraz doğrulama ortalaması en yüksek olsa da (%85.48), KNN tek ayrımda daha yüksek skor verdiği ve olasılık tahminlerinde daha kararlı olduğu için şampiyon model olarak seçilmeye devam etmektedir.

#### 3 Sınıflı Model (Normal / Disk Hernisi / Spondilolistezis)

Aynı 310 kayıt, bu kez Normal (100), Disk Hernisi (60) ve Spondilolistezis (150) olmak üzere 3 ayrı sınıfla etiketlenmiş orijinal UCI verisiyle (`column_3C.csv`) ayrıca eğitiliyor. Az örnekli Disk Hernisi sınıfının test setinde yeterince temsil edilmesi için train/test ayrımı `stratify=y` ile yapılıyor.

| Algoritma | Test Doğruluğu (tek ayrım) | 5-Katlı Çapraz Doğrulama |
| --- | --- | --- |
| **SVM (Destek Vektör)** 🏆 | **%83.87** | %85.48 (± %2.70) |
| Random Forest | %82.26 | %84.19 (± %4.38) |
| KNN (En Yakın Komşu) | %82.26 | %82.90 (± %2.41) |

Bu modda şampiyon model **SVM** oldu — 2 sınıflı modelden farklı bir algoritma seçilmesi, sınıf sayısı arttıkça (ve bir sınıf az örnekli olduğunda) hangi algoritmanın daha iyi genelleştirdiğinin değişebileceğini gösteriyor.

## Teknolojiler

| Katman | Teknoloji |
| --- | --- |
| Dil | Python |
| Makine öğrenmesi | scikit-learn (Random Forest, SVM, KNN) |
| Arayüz | Streamlit |
| Veri işleme / görselleştirme | pandas, matplotlib, seaborn |
| Model kaydı | joblib |

## Kurulum

Gereksinim: **Python 3.10+**

```bash
git clone https://github.com/mertokzlky/ortopedik-anomali-tespit.git
cd ortopedik-anomali-tespit
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Model dosyaları (`fiziktedavi_model.pkl`, `model_skorlari.pkl`, `fiziktedavi_model_3sinif.pkl`, `model_skorlari_3sinif.pkl`) depoda hazır geliyor; uygulamayı doğrudan çalıştırabilirsiniz:

```bash
streamlit run app.py
```

Modeli sıfırdan yeniden eğitmek isterseniz:

```bash
python model_egit.py
```

## Proje yapısı

```
ortopedik-anomali-tespit/
├── app.py                 # Streamlit arayüzü (tahmin + veri analizi sekmeleri)
├── model_egit.py          # Veriyi okuyup 3 algoritmayı eğiten, en iyisini kaydeden betik
├── toplu_tahmin.py        # CSV ile toplu tahmin için yardımcı fonksiyonlar (pytest ile test edilir)
├── column_2C.csv          # Vertebral Column veri seti (310 kayıt)
├── column_3C.csv          # Vertebral Column veri seti (3 sınıflı: Normal/Disk Hernisi/Spondilolistezis)
├── fiziktedavi_model.pkl  # Eğitilmiş en iyi model (KNN)
├── fiziktedavi_model_3sinif.pkl  # Eğitilmiş en iyi model (3 sınıflı, SVM)
├── model_skorlari.pkl     # Üç algoritmanın test başarı oranları
├── model_skorlari_3sinif.pkl     # Üç algoritmanın 3 sınıflı veri üzerindeki test başarı oranları
├── requirements.txt
├── .gitignore
└── docs/screenshots/      # README'deki ekran görüntüleri
```

## Dağıtım (Streamlit Community Cloud)

1. [share.streamlit.io](https://share.streamlit.io) adresine GitHub hesabınızla giriş yapın.
2. **New app** → bu repoyu seçin, ana dosya olarak `app.py`'yi belirtin.
3. Deploy edin; birkaç dakika içinde canlı bir link alırsınız. Bu linki yukarıdaki "Canlı demo" satırına ekleyin.

## Geliştirme fikirleri

- Görsel özelliklerin (banner, anatomi referans görseli) eklenmesi
- Tahmine hangi ölçümün ne kadar etki ettiğini gösteren açıklanabilirlik (feature importance) grafiği
- `pytest` ile modelin temel sağlık kontrollerinin (ör. bilinen bir girdi için beklenen sınıfı döndürmesi) test edilmesi
- GitHub Actions ile testlerin her push'ta otomatik çalıştırılması

## Sorumluluk reddi

Bu proje bir üniversite dersi kapsamında geliştirilmiş akademik bir çalışmadır ve **gerçek tıbbi teşhis amacıyla kullanılamaz**. Sonuçlar yalnızca eğitim ve gösterim amaçlıdır; kesin teşhis için mutlaka bir sağlık uzmanına başvurulmalıdır.

## Geliştirici

**Mert Ali Kızılkaya** — Yazılım Mühendisliği öğrencisi
GitHub: [@mertokzlky](https://github.com/mertokzlky)

Görsel Programlama dersi final projesi olarak geliştirilmiştir.
