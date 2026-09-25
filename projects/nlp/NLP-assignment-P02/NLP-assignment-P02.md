# Tugas Mandiri P02 — Pengujian Library Python untuk NLP

**Mata Kuliah**: Advanced Natural Language Processing  
**Pertemuan**: P02 — Python for NLP  
**Topik**: Eksplorasi dan pengujian library Python untuk NLP  
**Bentuk**: Praktik pemrograman dan laporan hasil pengujian

## 1. Latar Belakang

Python memiliki banyak library yang membantu proses Natural Language Processing (NLP), mulai dari tokenisasi dan analisis teks sampai machine learning dan deep learning. Setiap library memiliki tujuan, cara penggunaan, kelebihan, dan keterbatasan yang berbeda.

Pada tugas mandiri ini dilakukan pengujian sederhana terhadap library yang diperkenalkan dalam materi `Python for NLP Library`. Pengujian dilakukan untuk mengetahui ketersediaan library pada environment Python dan memastikan fungsi dasar masing-masing library dapat digunakan.

## 2. Tujuan

Tugas ini bertujuan untuk:

1. mengenali library Python yang umum digunakan dalam NLP;
2. menguji proses import setiap library;
3. menjalankan fungsi dasar dari library yang tersedia;
4. mencatat library yang berhasil, belum terpasang, atau mengalami error;
5. membandingkan penggunaan library untuk kebutuhan NLP.

## 3. Library yang Diuji

| No. | Library | Fokus Pengujian |
|---:|---|---|
| 1 | NLTK | Tokenisasi teks |
| 2 | scikit-learn | TF-IDF dan representasi fitur |
| 3 | Gensim | Pembuatan dictionary token |
| 4 | spaCy | Pemrosesan dokumen dan token |
| 5 | TextBlob | Analisis sentimen sederhana |
| 6 | fastText | Pemeriksaan API pelatihan teks |
| 7 | AllenNLP | Pemeriksaan package NLP |
| 8 | Polyglot | Pemeriksaan package NLP multibahasa |
| 9 | Hugging Face Transformers | Pemeriksaan pipeline Transformer |
| 10 | TensorFlow | Pembuatan model neural network sederhana |
| 11 | PyTorch | Pembuatan tensor |
| 12 | Keras | Pembuatan model neural network sederhana |
| 13 | NumPy | Operasi array dan komputasi numerik |
| 14 | pandas | Pengolahan dataset teks berbentuk tabel |
| 15 | Matplotlib / Seaborn | Visualisasi data dan hasil analisis |
| 16 | regex | Pencarian dan pembersihan pola teks |
| 17 | Hugging Face Datasets | Memuat dan mengelola dataset NLP |
| 18 | Hugging Face Tokenizers | Tokenisasi cepat untuk model Transformer |
| 19 | Sentence Transformers | Embedding kalimat dan pencarian semantik |
| 20 | langdetect | Deteksi bahasa teks |
| 21 | SciPy | Komputasi ilmiah dan sparse matrix |
| 22 | Jupyter | Eksperimen dan dokumentasi interaktif |
| 23 | XGBoost / LightGBM | Model boosting untuk klasifikasi teks |
| 24 | imbalanced-learn | Penanganan dataset tidak seimbang |
| 25 | Hugging Face Evaluate | Evaluasi model NLP |
| 26 | PEFT | Fine-tuning efisien seperti LoRA |
| 27 | Accelerate | Dukungan training CPU/GPU |
| 28 | BitsAndBytes | Quantization dan efisiensi model |
| 29 | Plotly | Visualisasi interaktif |
| 30 | Sastrawi | Stemming bahasa Indonesia |
| 31 | ROUGE / BLEU / BERTScore | Evaluasi ringkasan dan machine translation |
| 32 | FAISS | Pencarian kemiripan vector untuk semantic search |
| 33 | LangChain / LlamaIndex | Orkestrasi aplikasi LLM dan RAG |
| 34 | FastAPI | Penyediaan model NLP melalui REST API |
| 35 | MLflow | Pelacakan eksperimen dan model |

### Pembagian pengujian

#### Library wajib diuji di Google Colab

Program `NLP-assignment-P02.py` hanya menguji library berikut:

```text
NumPy, pandas, SciPy, regex, NLTK, spaCy, scikit-learn,
PyTorch, Transformers, Tokenizers, Datasets,
Sentence Transformers, langdetect, Sastrawi, Matplotlib, Seaborn
```

Library tersebut dipilih karena mewakili alur dasar NLP secara lengkap:
pengolahan data, pembersihan teks, tokenisasi, feature extraction, model
machine learning/deep learning, embedding, bahasa Indonesia, dan visualisasi.

#### Library yang tidak diuji pada tugas dasar

Library berikut hanya dijelaskan sebagai wawasan lanjutan dan tidak diwajibkan
dijalankan:

- **Gensim dan TextBlob** — berguna untuk eksperimen NLP klasik sederhana,
  tetapi tidak diperlukan untuk pipeline utama tugas ini;
- **fastText, AllenNLP, dan Polyglot** — memiliki penggunaan lebih khusus serta
  dependency atau dukungan yang tidak selalu konsisten di Colab;
- **TensorFlow dan Keras** — framework deep learning alternatif; PyTorch cukup
  digunakan sebagai framework utama dalam tugas ini;
- **XGBoost, LightGBM, dan imbalanced-learn** — diperlukan untuk pembanding
  supervised learning atau dataset tidak seimbang;
- **Evaluate, PEFT, Accelerate, dan BitsAndBytes** — digunakan untuk evaluasi,
  fine-tuning, distribusi training, dan quantization model besar;
- **FAISS, LangChain, dan LlamaIndex** — digunakan pada semantic search dan
  aplikasi RAG;
- **ROUGE, BLEU, dan BERTScore** — diperlukan untuk evaluasi khusus ringkasan,
  machine translation, atau generasi teks;
- **FastAPI dan MLflow** — digunakan untuk deployment dan tracking eksperimen,
  bukan untuk pengenalan dasar library NLP;
- **Plotly** — alternatif visualisasi interaktif, sedangkan Matplotlib dan
  Seaborn sudah cukup untuk tugas ini.

Library lanjutan dapat diuji pada tugas berikutnya ketika kebutuhan proyeknya
sudah jelas. Tidak mengujinya pada tugas ini bukan berarti library tersebut
tidak penting.

### Library tambahan yang wajib dipahami

Library berikut menjadi fondasi praktik NLP dan analisis data modern:

- **NumPy** — dasar komputasi array yang digunakan oleh banyak library machine learning;
- **pandas** — membaca, membersihkan, menggabungkan, dan menganalisis dataset;
- **Matplotlib/Seaborn** — membuat visualisasi distribusi data, label, dan hasil model;
- **regex** — membersihkan teks, mencari pola, dan mengekstraksi informasi;
- **Hugging Face Datasets** — mengakses dataset NLP dengan format yang konsisten;
- **Hugging Face Tokenizers** — mengubah teks menjadi token sebelum diproses Transformer;
- **Sentence Transformers** — membuat embedding untuk semantic search, clustering, dan similarity;
- **langdetect** — mendeteksi bahasa sebelum memilih pipeline NLP yang sesuai.
- **SciPy** — mendukung komputasi ilmiah, sparse matrix, dan operasi yang digunakan scikit-learn.
- **Jupyter** — menjalankan eksperimen, kode, visualisasi, dan catatan dalam satu notebook.
- **XGBoost/LightGBM** — model boosting yang berguna sebagai pembanding model klasifikasi teks.
- **imbalanced-learn** — menangani masalah label yang jumlah datanya tidak seimbang.
- **Evaluate** — menghitung metrik evaluasi NLP secara konsisten.
- **PEFT** — melakukan fine-tuning model besar dengan parameter yang lebih sedikit, misalnya LoRA.
- **Accelerate** — membantu menjalankan training pada CPU, satu GPU, atau beberapa GPU.
- **BitsAndBytes** — mengurangi kebutuhan memori melalui quantization model.
- **Plotly** — membuat grafik interaktif untuk eksplorasi dan presentasi hasil.
- **Sastrawi** — melakukan stemming dan pengolahan teks khusus bahasa Indonesia.
- **ROUGE/BLEU/BERTScore** — mengevaluasi ringkasan, terjemahan, dan kemiripan keluaran model.
- **FAISS** — mencari dokumen atau embedding yang paling mirip untuk semantic search dan RAG.
- **LangChain/LlamaIndex** — menghubungkan LLM dengan dokumen, tools, memory, dan vector store.
- **FastAPI** — mengubah pipeline NLP menjadi layanan API yang dapat dipanggil aplikasi lain.
- **MLflow** — mencatat parameter, metrik, artefak, dan versi model selama eksperimen.

Pipeline konsep tugas:

```text
dataset → pandas/NumPy → regex → tokenizer → model → evaluasi → visualisasi
```

### Prioritas pembelajaran

Library tidak harus dipasang sekaligus. Urutan yang disarankan:

1. **Wajib/fondasi**: NumPy, pandas, regex, Jupyter, NLTK atau spaCy, dan scikit-learn.
2. **Wajib untuk NLP modern**: PyTorch, Transformers, Tokenizers, Datasets, dan Sentence Transformers.
3. **Penting untuk riset dan produksi**: Evaluate, PEFT, Accelerate, dan BitsAndBytes.
4. **Pendukung dan pembanding**: SciPy, XGBoost/LightGBM, imbalanced-learn, Plotly, TextBlob, dan Gensim.
5. **Spesialisasi**: Sastrawi untuk bahasa Indonesia, ROUGE/BLEU/BERTScore untuk evaluasi generatif, FAISS untuk vector search, LangChain/LlamaIndex untuk RAG, FastAPI untuk deployment, dan MLflow untuk tracking eksperimen.

Klasifikasi ini membedakan library yang perlu dipahami konsep dasarnya dari
library lanjutan yang digunakan ketika kebutuhan proyek sudah lebih spesifik.

## 4. Program yang Dibuat

Program pengujian berada pada file:

```text
tugas/NLP-assignment-P02.py
```

Program menggunakan tiga status hasil:

- `PASS`: library berhasil di-import dan fungsi dasarnya berhasil dijalankan;
- `SKIP`: library belum terpasang pada environment;
- `FAIL`: library terpasang tetapi pengujian mengalami kesalahan.

Pengujian dibuat ringan dan tidak mengunduh model bahasa berukuran besar.

## 5. Cara Menjalankan di Google Colab

Seluruh pengujian tugas ini dilakukan di **Google Colab**, bukan di Ubuntu
lokal. Buat notebook baru di Colab, lalu jalankan cell berikut.

### 5.1 Menghubungkan repository

```python
!git clone https://github.com/USERNAME/REPOSITORY.git
%cd REPOSITORY
```

Ganti `USERNAME/REPOSITORY` dengan alamat repository GitHub yang berisi folder
`SMTIII-ADVANCED NLP/P02-Python-for-NLP`.

### 5.2 Memasang dependency di Colab

```python
!pip install -q nltk spacy transformers datasets tokenizers \
    sentence-transformers langdetect Sastrawi seaborn regex
```

Library bawaan Colab seperti NumPy, pandas, SciPy, Matplotlib, scikit-learn,
PyTorch, dan TensorFlow tidak perlu dipasang ulang kecuali versi tertentu
memang diperlukan.

### 5.3 Menjalankan program pengujian

```python
!python "SMTIII-ADVANCED NLP/P02-Python-for-NLP/tugas/NLP-assignment-P02.py"
```

Jika posisi notebook sudah berada di folder P02, gunakan:

```python
!python tugas/NLP-assignment-P02.py
```

Contoh output:

```text
[PASS] scikit-learn     - TF-IDF menghasilkan matriks (2, 5)
[SKIP] NLTK             - belum terpasang (nltk)
================================================
Ringkasan: 1 PASS, 1 SKIP, 0 FAIL
```

Jumlah hasil dapat berbeda bergantung pada versi runtime dan dependency yang
tersedia di Google Colab.

## 6. Prosedur Pengujian

1. Membuka Google Colab dan menghubungkan repository GitHub.
2. Memasang dependency yang diperlukan pada runtime Colab.
3. Menentukan teks pendek sebagai data uji.
4. Mengimpor library menggunakan Python.
5. Menjalankan satu fungsi dasar yang mewakili kegunaan library.
6. Memeriksa hasil dengan assertion sederhana.
7. Mencatat status dan keterangan hasil pengujian.
8. Membandingkan library berdasarkan fungsi dan kemudahan penggunaannya.

## 7. Analisis yang Harus Ditulis

Setelah program dijalankan, buat analisis berdasarkan pertanyaan berikut:

1. Library apa saja yang berhasil dijalankan?
2. Library apa saja yang belum terpasang?
3. Apa penyebab library tertentu berstatus `SKIP` atau `FAIL`?
4. Library mana yang paling mudah digunakan untuk pemrosesan teks dasar?
5. Apa perbedaan penggunaan NLTK, spaCy, dan Transformers?
6. Mengapa scikit-learn tetap penting dalam proyek NLP?
7. Kapan sebaiknya menggunakan PyTorch, TensorFlow, atau Keras?
8. Library mana yang paling sesuai untuk proyek NLP yang akan dikembangkan?

## 8. Kesimpulan

Tuliskan kesimpulan berdasarkan hasil pengujian. Kesimpulan minimal memuat jumlah library berstatus `PASS`, `SKIP`, dan `FAIL`, library yang paling sesuai untuk kebutuhan dasar NLP, serta kendala instalasi atau kompatibilitas yang ditemukan.

## 9. Berkas Pengumpulan

Berkas tugas mandiri terdiri atas:

1. `NLP-assignment-P02.md` — penjelasan dan laporan tugas;
2. `NLP-assignment-P02.py` — program pengujian;
3. tangkapan layar atau salinan output cell Google Colab;
4. PDF hasil export Markdown, jika diminta oleh dosen.

## 10. Catatan Teknis

Pengujian hanya dilakukan di Google Colab. Tidak perlu menjadikan hasil
environment Ubuntu lokal sebagai bagian dari laporan. Beberapa library
memiliki dependency besar atau memerlukan model tambahan. Status `SKIP` berarti
library belum tersedia pada runtime Colab, bukan berarti program pengujian
gagal.
