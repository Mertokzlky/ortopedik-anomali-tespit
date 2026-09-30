import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os

OZELLIK_ISIMLERI = [
    'Pelvik_İnsidans',
    'Pelvik_Eğim',
    'Lumbar_Lordoz_Açısı',
    'Sakral_Eğim',
    'Pelvik_Yarıçap',
    'Spondilolistezis_Derecesi',
]


def modeli_egit(dosya_adi, model_dosya_adi, skor_dosya_adi, baslik, stratify=False):
    """Bir veri setini okuyup 3 algoritmayı eğitir, çapraz doğrulama yapar
    ve en başarılı modeli diske kaydeder. Hem 2 sınıflı (column_2C.csv)
    hem de 3 sınıflı (column_3C.csv) veri seti için kullanılır.

    stratify=True: az örnekli bir sınıf (ör. 3 sınıflı veri setindeki
    60 kişilik Disk Hernisi grubu) test/eğitim ayrımında dengeli
    dağılsın diye kullanılır. 2 sınıflı model için mevcut sonuçlarla
    (README'deki skorlar) tutarlılığı korumak amacıyla kapalı bırakıldı.
    """

    print(f"\n{'=' * 50}")
    print(f"{baslik}")
    print(f"{'=' * 50}")

    if not os.path.exists(dosya_adi):
        print(f"HATA: '{dosya_adi}' dosyası klasörde bulunamadı! Lütfen ismini kontrol et.")
        return

    df = pd.read_csv(dosya_adi)
    df.columns = OZELLIK_ISIMLERI + ['Durum']

    # VERİYİ HAZIRLA
    X = df.drop('Durum', axis=1)
    y = df['Durum']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if stratify else None)

    # MODELLERİ EĞİT
    modeller = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (Destek Vektör)": SVC(probability=True),
        "KNN (En Yakın Komşu)": KNeighborsClassifier(n_neighbors=5)
    }

    sonuclar = {}

    print(f"Toplam {len(df)} kayıt üzerinde eğitim yapılıyor "
          f"({y.nunique()} sınıf: {', '.join(sorted(y.unique()))})...")

    for isim, model in modeller.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        basari = accuracy_score(y_test, y_pred)
        sonuclar[isim] = basari
        print(f"👉 {isim} Başarısı: %{basari * 100:.2f}")

    # ÇAPRAZ DOĞRULAMA (5 katlı, her katta sınıf oranı korunur)
    # Tek bir train/test ayrımı şansa bağlı olabilir; bu yüzden tüm veri
    # 5 farklı şekilde bölünüp her model 5 kez test edilir.
    print("\n📊 5-Katlı Çapraz Doğrulama (tüm veri üzerinde):")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for isim, model in modeller.items():
        kat_skorlari = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        print(f"👉 {isim}: %{kat_skorlari.mean() * 100:.2f} (± %{kat_skorlari.std() * 100:.2f})")

    en_iyi_model_ismi = max(sonuclar, key=sonuclar.get)
    en_iyi_model = modeller[en_iyi_model_ismi]

    print(f"\n🏆 ŞAMPİYON MODEL: {en_iyi_model_ismi}")

    joblib.dump(en_iyi_model, model_dosya_adi)
    joblib.dump(sonuclar, skor_dosya_adi)
    print(f"💾 Model → {model_dosya_adi}, skorlar → {skor_dosya_adi} olarak kaydedildi!")


if __name__ == "__main__":
    print("Model eğitimi başlıyor...")

    modeli_egit(
        dosya_adi="column_2C.csv",
        model_dosya_adi="fiziktedavi_model.pkl",
        skor_dosya_adi="model_skorlari.pkl",
        baslik="2 SINIFLI MODEL (Normal / Anormal)",
        stratify=False,
    )

    modeli_egit(
        dosya_adi="column_3C.csv",
        model_dosya_adi="fiziktedavi_model_3sinif.pkl",
        skor_dosya_adi="model_skorlari_3sinif.pkl",
        baslik="3 SINIFLI MODEL (Normal / Disk Hernisi / Spondilolistezis)",
        stratify=True,
    )

    print("\n✅ Tüm modeller eğitildi.")
