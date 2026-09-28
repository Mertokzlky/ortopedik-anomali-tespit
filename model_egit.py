import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os

dosya_adi = "column_2C.csv"

print("Model eğitimi başlıyor...")

if os.path.exists(dosya_adi):
    df = pd.read_csv(dosya_adi)
    
    df.columns = [
        'Pelvik_İnsidans', 
        'Pelvik_Eğim', 
        'Lumbar_Lordoz_Açısı', 
        'Sakral_Eğim', 
        'Pelvik_Yarıçap', 
        'Spondilolistezis_Derecesi', 
        'Durum'
    ]
    
    # VERİYİ HAZIRLA
    X = df.drop('Durum', axis=1)
    y = df['Durum']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # MODELLERİ EĞİT
    modeller = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (Destek Vektör)": SVC(probability=True),
        "KNN (En Yakın Komşu)": KNeighborsClassifier(n_neighbors=5)
    }

    sonuclar = {}
    
    print(f"Toplam {len(df)} kayıt üzerinde eğitim yapılıyor...")
    
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
    
    joblib.dump(en_iyi_model, 'fiziktedavi_model.pkl')
    joblib.dump(sonuclar, 'model_skorlari.pkl')
    print("💾 Model ve skorlar başarıyla kaydedildi!")
    
else:
    print(f"HATA: '{dosya_adi}' dosyası klasörde bulunamadı! Lütfen ismini kontrol et.")