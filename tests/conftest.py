# Dosya adı: tests/conftest.py
# Tüm test dosyalarının ortak kullandığı fixture'lar burada tanımlanır.
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

# Proje kökünü import path'ine ekle (toplu_tahmin.py'yi bulabilsin diye)
KOK_DIZIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK_DIZIN))

OZELLIKLER = [
    'Pelvik_İnsidans',
    'Pelvik_Eğim',
    'Lumbar_Lordoz_Açısı',
    'Sakral_Eğim',
    'Pelvik_Yarıçap',
    'Spondilolistezis_Derecesi',
]


@pytest.fixture(scope="session")
def kok_dizin():
    return KOK_DIZIN


@pytest.fixture(scope="session")
def model_2sinif(kok_dizin):
    return joblib.load(kok_dizin / "fiziktedavi_model.pkl")


@pytest.fixture(scope="session")
def model_3sinif(kok_dizin):
    return joblib.load(kok_dizin / "fiziktedavi_model_3sinif.pkl")


@pytest.fixture(scope="session")
def veri_2sinif(kok_dizin):
    df = pd.read_csv(kok_dizin / "column_2C.csv")
    df.columns = OZELLIKLER + ["Durum"]
    return df


@pytest.fixture(scope="session")
def veri_3sinif(kok_dizin):
    df = pd.read_csv(kok_dizin / "column_3C.csv")
    df.columns = OZELLIKLER + ["Durum"]
    return df
