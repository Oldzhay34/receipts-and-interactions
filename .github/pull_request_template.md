## Aşama
<!-- ör. 1.3 — Eksik veri yönetimi | Sahip: A | Branch: a/1.3-missing -->

## Özet
<!-- Ne yapıldı? 2-4 madde. -->

## Kontrol listesi
- [ ] **Sahiplik:** `git diff --name-only main` sadece bu aşamanın sahip olduğu dosyaları gösteriyor (çıktıyı aşağıya yapıştır)
- [ ] **Arayüzler:** Başka bir modülün public arayüzünü değiştirmedim (gerekiyorsa sahibine sordum)
- [ ] **Testler:** `pytest -q` geçiyor (fixtures + data/sample)
- [ ] **Biçim:** `ruff format --check` ve `ruff check` temiz
- [ ] **Colab:** Notebook standart Colab hücresiyle başlıyor, sadece `src/recsys` çağırıyor, tam veriyle Colab'da çalıştırıldı (commit: `_______`)
- [ ] **Manifest:** Tüm çıktılar ArtifactStore ile yazıldı; yeni sürüm(ler): `outputs/<aşama>/vN`; şema doğrulaması var
- [ ] **Sayıların kaynağı:** Rapor/yorum/docstring/commit'teki her sayının kaynağı bir çıktı dosyası
- [ ] **Leakage:** Fit edilen her şey sadece train'den öğreniliyor (veya bu PR'da fit yok)
- [ ] **Secret:** Token, şifre, .env commit edilmedi

## `git diff --name-only main`
```
```

## İnceleyen (diğer kişi)
- [ ] Kodu okudum, testleri yerelde çalıştırdım
- [ ] Onaylıyorum
