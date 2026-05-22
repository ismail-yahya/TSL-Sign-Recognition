--- START OF FILE Paste May 14, 2026 - 11:37PM ---

<div align="center">

# T.C.
## NECMETTİN ERBAKAN ÜNİVERSİTESİ
## SEYDİŞEHİR AHMET CENGİZ MÜHENDİSLİK FAKÜLTESİ

<br>

### BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ

<br><br><br>

# İŞARET DİLİNİ METİN VE KONUŞMAYA DÖNÜŞTÜRME SİSTEMİ 

<br>

**22370031804-Hasan ELRECEB**

**22370031805-İsmail YAHYA**

<br><br>

**Danışman:**
Dr. Öğr. Üyesi Yunus Emre Göktepe

<br><br>

**MÜHENDİSLİK PROJESİ II** 

<br><br><br>

**Mart - 2026**
**SEYDİŞEHİR** 
**Her Hakkı Saklıdır**

</div>

---

# LİSANS BİTİRME ÖDEVİ SONUÇ FORMU

………………………………. tarafından ……………………………….. danışmanlığında hazırlanan “……………………………………………..” başlıklı lisans bitirme ödevi tarafımızdan incelenmiş, kapsamı ve niteliği açısından, ... / … / 20… tarihinde bir Lisans Bitirme Ödevi olarak kabul edilmiştir / edilmemiştir.

<br>

**Unvan Ad SOYAD**
Danışman

<br>

**Unvan Ad SOYAD**
Jüri Üyesi

<br>

**Unvan Ad SOYAD**
Jüri Üyesi

<br>

**Unvan Ad SOYAD**
Bölüm Başkanı

---

# TEZ BİLDİRİMİ

Bu tezdeki bütün bilgilerin etik davranış ve akademik kurallar çerçevesinde elde edildiğini ve tez yazım kurallarına uygun olarak hazırlanan bu çalışmada bana ait olmayan her türlü ifade ve bilginin kaynağına eksiksiz atıf yapıldığını bildiririm.

<br>

# DECLARATION PAGE

I hereby declare that all information in this document has been obtained and presented in accordance with academic rules and ethical conduct. I also declare that, as required by these rules and conduct, I have fully cited and referenced all material and results that are not original to this work.

<br>

Seydişehir, …./…./ 20…

<br>

**İmza**
Hasan ELRECEB

**İmza**
İsmail YAHYA

---

# ÖZET

**İŞARET DİLİNİ METİN VE KONUŞMAYA DÖNÜŞTÜRME SİSTEMİ**

Bu çalışmada, Türk İşaret Dili (TİD) için yapay zekâ tabanlı، gerçek zamanlı ve uçtan uca bir çeviri sistemi geliştirilmiştir. Projenin temel motivasyonu, işitme engelli bireylerin toplumsal entegrasyonunu engelleyen iletişim bariyerlerini, modern bilgisayarlı görü ve derin öğrenme teknikleriyle minimize etmektir. Literatürdeki ağırlıklı olarak görsel (RGB) verilere dayanan ve yüksek hesaplama gücü gerektiren modellerin aksine; bu çalışmada، MediaPipe Holistic aracılığıyla elde edilen optimize edilmiş iskeletsel anahtar noktalar (landmarks) kullanılarak hafif ve verimli bir mimari oluşturulmuştur.

Yöntem kapsamında, zamansal ve uzamsal (spatiotemporal) dinamikleri geleneksel ardışık modellerden (CNN, RNN) daha etkili bir şekilde modelleyebilen "Sadece Kodlayıcı" (Encoder-only) tabanlı Transformer mimarisi tercih edilmiştir. AUTSL veri setinde yer alan 226 izole işaret sınıfı üzerinde eğitilen model, test setinde %90.96 genel doğruluk (Accuracy) ve %98.26 ilk-5 (Top-5) doğruluk oranına ulaşmıştır. Sistemin pratik uygulanabilirliğini artırmak amacıyla, tanınan işaretlerin anlık olarak sese dönüştürülmesi için Piper TTS motoru entegre edilmiş ve düşük gecikmeli (low-latency) bir çalışma performansı elde edilmiştir.

Elde edilen sonuçlar, önerilen iskelet tabanlı Transformer modelinin Türk İşaret Dili tanıma görevlerinde yüksek doğruluk ve genelleme yeteneği sunduğunu kanıtlamaktadır. Ayrıca bu çalışma, Birleşmiş Milletler Sürdürülebilir Kalkınma Amaçları’ndan Nitelikli Eğitim (SKA 4), Sanayi, Yenilikçilik ve Altyapı (SKA 9) ve Eşitsizliklerin Azaltılması (SKA 10) hedefleriyle doğrudan örtüşerek kapsayıcı bir teknolojik çözüm sunmaktadır.

**Anahtar Kelimeler:** Türk İşaret Dili, Transformer, MediaPipe, Derin Öğrenme, Metinden Konuşmaya Çeviri (TTS), Bilgisayarlı Görü.

---

# ABSTRACT

**SIGN LANGUAGE TO TEXT AND SPEECH CONVERSION SYSTEM**

In this study, an artificial intelligence-based, real-time, end-to-end translation system for Turkish Sign Language (TİD) is developed. The primary motivation of the project is to minimize the communication barriers that hinder the social integration of hearing-impaired individuals by leveraging modern computer vision and deep learning techniques. In contrast to existing literature models that predominantly rely on raw visual (RGB) data and require high computational power, this research implements a lightweight and efficient architecture utilizing optimized skeletal landmarks extracted via MediaPipe Holistic.

The methodology employs an "Encoder-only" Transformer architecture, which models spatiotemporal dynamics more effectively than traditional sequential models (CNN, RNN). Trained on 226 isolated sign classes from the Ankara University Turkish Sign Language (AUTSL) dataset, the model attained an overall accuracy of 90.96% and a Top-5 accuracy of 98.26% on the test set. To enhance practical applicability, the Piper TTS engine was integrated for instantaneous text-to-speech conversion, achieving high-fidelity and low-latency performance.

The experimental results demonstrate that the proposed skeletal-based Transformer model offers high accuracy and robust generalization capabilities for Turkish Sign Language recognition tasks. Furthermore, this study provides an inclusive technological solution that directly aligns with the United Nations Sustainable Development Goals (SDGs), specifically Quality Education (SDG 4), Industry, Innovation, and Infrastructure (SDG 9), and Reduced Inequalities (SDG 10).

**Keywords:** Turkish Sign Language, Transformer, MediaPipe, Deep Learning, Text-to-Speech (TTS), Computer Vision.

---

# ÖNSÖZ

Bu projede, yapay zekâ tekniklerini kullanarak işaret dilini metin ve konuşmaya dönüştüren bir sistem geliştirilmiştir. Projenin çıkış noktası, işitme engelli bireylerle toplum arasındaki iletişimi kolaylaştırmak ve onların günlük yaşama entegrasyonunu artırmak için yenilikçi bir teknolojik çözüm ortaya koyma ihtiyacıdır. Bu sistem, işaret dilini herkes tarafından anlaşılabilir yazılı ve sözlü bir forma dönüştürmeyi hedeflemektedir.

Bu proje, işitme engelli bireyler ile toplum arasında iletişim köprüsü kurma yolunda önemli bir adım niteliğindedir. Projede Derin Öğrenme (Deep Learning), Bilgisayarlı Görü (Computer Vision), Doğal Dil İşleme (NLP) ve Metinden Konuşmaya Dönüştürme (TTS) gibi ileri teknolojiler kullanılarak bütünleşik bir etkileşim deneyimi oluşturulmuştur.

Öncelikle, bizlere araştırma ve yenilik yapma fırsatları sunan Necmettin Erbakan Üniversitesi’ne en içten teşekkürlerimi sunarım. Proje süresince değerli bilgi, rehberlik ve desteğini bizden esirgemeyen danışman hocamız Dr. Öğr. Üyesi Yunus Emre Göktepe’ye şükranlarımı sunarım.

Bu süreçte daima yanımda olan, bana moral, destek ve güven sağlayarak akademik ilerleyişime katkıda bulunan aileme en derin minnettarlığımı ifade etmek isterim. Onların sabrı, desteği ve teşviki bu aşamaya ulaşmamda en önemli güç kaynağım olmuştur.

Bu projenin, yapay zekânın toplumsal fayda odaklı kullanımına katkıda bulunmasını, işitme engelli bireylerin yaşam kalitesini artırmasını ve modern teknolojilerin insan iletişiminde daha kapsayıcı bir rol üstlenmesine zemin hazırlamasını temenni ediyorum.

**22370031804 - Hasan ELRECEB**
**22370031805 - İsmail YAHYA**
SEYDİŞEHİR - 2026

---

# İÇİNDEKİLER

- ÖZET iv
- ABSTRACT v
- ÖNSÖZ vi
- İÇİNDEKİLER vii
- ŞEKİLLER LİSTESİ ix
- ÇİZELGELER/TABLOLAR LİSTESİ x
- SİMGELER VE KISALTMALAR xi
- 1. GİRİŞ 1
- 2. KAYNAK ARAŞTIRMASI 4
  - 2.1. İşaret Dili Tanıma Sistemlerine Genel Bakış 4
  - 2.2. Sensör Tabanlı İşaret Dili Tanıma Sistemleri 4
  - 2.3. Görsel (Kamera) Tabanlı İşaret Dili Tanıma Sistemleri 5
  - 2.4. Gerçek Zamanlı (Real-Time) İşaret Dili Tanıma Uygulamaları 6
  - 2.5. İşaret Dili Çeviri ve Seslendirme Sistemleri 6
  - 2.6. Genel Değerlendirme ve Literatürdeki Boşluklar 7
- 3. MATERYAL VE YÖNTEM 8
  - 3.1. Çalışma Ortamı ve Donanım/Yazılım Araçları 8
    - 3.1.1. Yazılım araçları 8
    - 3.1.2. Donanım araçları ve hesaplama platformu 9
  - 3.2. Veri Seti Tanımı ve Veri Yapısı 9
    - 3.2.1. Veri setinin kapsamı ve istatistiksel özellikleri 9
    - 3.2.2. Teknik spesifikasyonlar ve çevresel zorluklar 10
    - 3.2.3. Proje kapsamında veri kullanımı ve landmark çıkarımı 10
    - 3.2.4. Değerlendirme protokolü 11
  - 3.3. MediaPipe Holistic ile İskeletsel Öznitelik Çıkarımı 11
    - 3.3.1. Optimize edilmiş landmark seçimi ve veri sadeleştirme 11
    - 3.3.2. Koordinat sistemi ve öznitelik vektörü oluşturma 12
  - 3.4. Veri Ön İşleme ve Normalizasyon 12
    - 3.4.1. Uzamsal normalizasyon 13
    - 3.4.2. Zamansal hizalama: padding ve masking 13
    - 3.4.3. Veri temizleme ve kalite kontrol 14
  - 3.5. Transformer Tabanlı Zamansal Modelleme 14
    - 3.5.1. Kodlayıcı Temelli Mimari ve Özellik Projeksiyonu 15
    - 3.5.2. Öğrenilebilir Konumsal Gömme (Positional Embedding) 15
    - 3.5.3. Çok Başlı Öz-Dikkat (Multi-Head Self-Attention) Mekanizması 15
    - 3.5.4. Sınıflandırma ve Optimizasyon Stratejisi 15
  - 3.6. Metin ve Konuşma Dönüştürme Süreci 17
    - 3.6.1. Sınıflandırma sonuçlarının metinleştirilmesi ve çıkarım istikrarı 17
    - 3.6.2. Metinden konuşmaya çeviri (Piper TTS) entegrasyonu 17
    - 3.6.3. Sistem çıktısının bütünleştirilmesi 17
- 4. ARAŞTIRMA SONUÇLARI VE TARTIŞMA 19
  - 4.1. Eğitim Süreci ve Hiperparametre Optimizasyonu 19
  - 4.2. Modelin Test Seti Üzerindeki Başarısı 20
  - 4.3. Sınıf Bazlı Performans ve Karışıklık Matrisi Analizi 21
  - 4.4. Gerçek Zamanlı Çıkarım ve Entegrasyon Performansı 21
  - 4.5. Literatür ile Karşılaştırma ve Tartışma 21
- 5. SONUÇLAR VE ÖNERİLER 23
  - 5.1 Sonuçlar 23
  - 5.2 Öneriler 23
    - 5.2.1. Kullanıcı arayüzü geliştirilmesi ve entegrasyonu 24
    - 5.2.2. Model optimizasyonu ve performans iyileştirme 24
    - 5.2.3. Zorlu sınıflar için çoklu modalite yaklaşımı 24
    - 5.2.4. Sürekli işaret dili çevirisine geçiş 24
- 6. KAYNAKLAR 25
- EKLER 27

---

# ŞEKİLLER LİSTESİ

- Şekil 1: Sistemin Genel İş Akış Şeması (İşaret Dili - Landmark - Transformer - TTS).
- Şekil 2: Veri Setindeki Farklı Arka Plan, Işıklandırma ve Konum Çeşitliliğini Gösteren Örnek Görüntüler.
- Şekil 3: MediaPipe Tarafından Sağlanan Tüm Ham İşaret Noktaları (Sol) ile Model İçin Seçilen Optimize Edilmiş Noktaların (Sağ) Karşılaştırılması.
- Şekil 4: Ham İşaret Noktalarından (Sol) Vücut Merkezli Normalizasyon ile Standartlaştırılmış Veriye (Sağ) Geçiş Süreci.
- Şekil 5: Önerilen Transformer (Encoder-only) Model Mimarisi ve Veri Akış Diyagramı. (Yapay zeka desteği ile tasarlanmıştır.)
- Şekil 6: Model Çıktısından Ses Sentezine Geçiş Şeması.
- Şekil 7: İşaret Dilinden Konuşmaya Dönüşüm Sisteminin Uçtan Uca Çalışma Mimarisi ve Aşamalı Veri Akış Şeması (Yapay zeka desteği ile tasarlanmıştır).
- Şekil 8: Transformer modelinin 200 epokluk eğitim sürecinde elde edilen eğitim (mavi) ve doğrulama (kırmızı) setlerine ait doğruluk (sol) ve kayıp (sağ) değişim eğrileri.

---

# ÇİZELGELER/TABLOLAR LİSTESİ

- Çizelge 1.1. Bir 10

---

# SİMGELER VE KISALTMALAR

**SİMGELER**

**KISALTMALAR**

- **AI** Artificial Intelligence
- **ASL** American Sign Language
- **CER** Character Error Rate
- **CNN** Convolutional Neural Network
- **DCT** Discrete Cosine Transform
- **DNN** Deep Neural Network
- **HMM** Hidden Markov Model
- **IMU** Inertial Measurement Unit
- **ISL** Indian Sign Language
- **iSTFT** Inverse Short-Time Fourier Transform
- **ML** Machine Learning
- **MNMT** Multimodal Neural Machine Translation
- **NLP** Natural Language Processing
- **OpenCV** Open Source Computer Vision Library
- **RNN** Recurrent Neural Network
- **SKA** Sürdürülebilir Kalkınma Amaçları
- **SLR** Sign Language Recognition
- **TD-Net** Triple Feature Double Motion Network
- **TİD** Türk İşaret Dili
- **TTS** Text-to-Speech
- **VITS** Variational Inference Text-to-Speech
- **WER** Word Error Rate

---

# 1. GİRİŞ

Günümüzde teknolojik gelişmeler özellikle yapay zekâ, derin öğrenme ve bilgisayarlı görü alanlarında büyük bir ivme kazanmıştır. Bu gelişmeler sayesinde makineler artık insan davranışlarını analiz edebilmekte, yorumlayabilmekte ve anlamlandırabilmektedir. Yapay zekâ sistemleri yalnızca endüstriyel uygulamalarda değil, aynı zamanda sosyal yaşamı kolaylaştırmak amacıyla da yoğun şekilde kullanılmaya başlanmıştır. Toplumsal fayda odaklı projelerde yapay zekâ uygulamaları giderek yaygınlaşmakta ve engelli bireylerin yaşam kalitesini artırmaya yönelik çözümler geliştirilmektedir.

İşitme engelli bireyler için iletişim, günlük yaşamda en temel ihtiyaçlardan biridir; ancak toplumun büyük bir kısmının işaret dilini bilmemesi nedeniyle bu bireyler çoğu zaman kendilerini ifade etme konusunda ciddi zorluklarla karşılaşmaktadır. Eğitim, sağlık hizmetleri, kamu kurumlarıyla iletişim ve sosyal etkileşim gibi alanlarda duyulan iletişim ihtiyacı, işaret dilini bilen kişinin her zaman bulunamamasından dolayı kesintiye uğramaktadır. Bu sorunun çözümü için geliştirilen işaret dilini sesli konuşmaya dönüştüren bir sistem, işitme engelli bireylerin bağımsızlığını artırma potansiyeline sahiptir.

Bu çalışmada geliştirilen sistem, Türk İşaret Dilinde yapılan el ve vücut hareketlerini kameradan algılayarak bunları metne ve daha sonra sese dönüştürmeyi amaçlamıştır. Böyle bir sistemin geliştirilmesi; hem toplumda iletişim engellerinin azaltılması hem de dijital dönüşümün insana dokunan bir alana taşınması bakımından son derece değerlidir. Yapay zekâ destekli çözümlerle işaret dilinin anlaşılabilir bir forma dönüştürülmesi, işitme engelli bireylerin toplumsal hayata aktif katılımını desteklemekte ve iletişim eşitliği sağlamaktadır.

### Araştırmanın Amacı

Bu araştırmanın temel amacı, Türk İşaret Dilinde kullanılan belirli el ve vücut hareketlerini algılayan, bu hareketlerden anlam çıkarabilen ve elde edilen çıktıları metin ve ses formatına dönüştürebilen uçtan uca bir yapay zekâ sistemi geliştirmek olmuştur. Sistem, bilgisayarlı görü algoritmalarıyla (MediaPipe) işaretleri iskeletsel verilere dönüştürür, Transformer tabanlı bir mimari ile analiz ederek anlam karşılıklarını metin hâline getirir ve Piper TTS (Text-to-Speech) teknolojisi ile doğal ses çıktısı üretir. Böylece kullanıcı kameraya karşı el işaretleriyle bir kelime oluşturabilir ve sistem bunu anında konuşma hâline getirerek karşı tarafa iletebilir.

Çalışma kapsamında geliştirilen sistem, Ankara Üniversitesi Türk İşaret Dili (AUTSL) veri setinde yer alan ve günlük hayatta sık kullanılan 226 farklı işaret sınıfını tanıyacak şekilde eğitilmiştir. Harfler ve sayılarla sınırlı kalmayan bu geniş sözcük dağarcığı, sistemin ilk aşamada güçlü bir prototip hâline getirilmesini sağlamış ve uygulanabilirliğini büyük ölçüde artırmıştır. Yüz ifadeleri, duygu aktarımı veya bağlama göre anlam yorumlama gibi ileri seviye analizler ise işaret dilinin karmaşık yapısı nedeniyle bu çalışmanın kapsamı dışında tutulmuştur.

### Araştırmanın Önemi

Türk İşaret Dili üzerine yapılan teknolojik projeler oldukça sınırlıdır. Literatürde mevcut çalışmaların çoğu Amerikan veya İngiliz işaret dilleri üzerine yoğunlaşmıştır ve Türk İşaret Diline yönelik açık veri kaynakları yakın zamana kadar yeterli değildi. Bu çalışma, Türk İşaret Diline özgü geniş kapsamlı bir veri seti (AUTSL) kullanarak yerli bir Transformer modeli geliştirilmesi açısından önemlidir.

Projenin özgün değeri, görsel işaretlerin iskeletsel çıkarım yoluyla analiz edilmesi, anlamlı metne dönüştürülmesi ve metnin doğal bir sesle ifade edilmesi gibi ayrı disiplinlerin başarılı bir şekilde birleştirilmesinden gelmektedir. Yapay zekâ, bilgisayarlı görü ve doğal dil işleme alanlarının entegre şekilde kullanılması, sistemin hem akademik hem de sosyal etkisini artırmaktadır. Sistem günlük hayatta işitme engelli bireylere destek sağlayarak, onların iletişim süreçlerindeki bağımlılıklarını azaltabilir ve özgüvenlerini artırabilir.

### Veri Kaynakları ve Yöntem

Çalışmada veri kaynağı olarak, literatürde geniş kabul gören ve Kaggle platformu üzerinden erişilen AUTSL (Ankara University Turkish Sign Language) veri seti kullanılmıştır. Bu veri setinin içeriği, farklı ortam, arka plan ve kişiler tarafından kaydedilen on binlerce izole işaret videosundan oluşmaktadır ve bu durum modelin genel performansını olumlu yönde etkilemektedir.

Sistem; işaret görüntülerinin MediaPipe kütüphanesi ile analiz edilerek uzamsal iskelet (landmark) verilerinin çıkarılması, bu zaman serisi verilerinin Transformer (Encoder-only) derin öğrenme algoritmalarıyla sınıflandırılması süreçlerini kapsamaktadır. Son aşamada ise üretilen metin, Piper TTS motoru kullanılarak gerçek zamanlı sesli çıktıya dönüştürülmektedir. Bu işlem sırası sayesinde işaret dili konuşmaya dönüştürülmekte ve anında iletişim sağlanabilmektedir.

### Sınırlılıklar

Bu çalışmada geliştirilen sistem, AUTSL veri setinde yer alan 226 farklı işaret sınıfı ile sınırlandırılmıştır. Sistem bu kelimeleri yüksek doğrulukla tanıyabilse de, yüz ifadeleri, duygu aktarımı ve bağlama göre anlam çıkarımı gibi karmaşık işaret yapıları kapsamda değildir. Ayrıca, modelin "izole işaret tanıma" (isolated sign recognition) prensibiyle çalışması nedeniyle, işaretlerin kelime kelime ve aralarında belirgin boşluklar bırakılarak yapılması gerekmektedir; kesintisiz ve akıcı cümle çevirisi (continuous sign language translation) şu an için sistemin sınırları dışındadır.

Kullanılan veri setlerindeki video kaliteleri, kamera açıları ve işaret hızları arasındaki farklılıklar, modelin performansını zaman zaman etkileyebilmektedir. Sistem gerçek zamanlı olarak çalışabilse de, çok düşük ışık koşulları veya el işaretlerinin kameranın görüş açısı dışında (çok hızlı) yapılması gibi durumlar doğruluk oranını düşürebilir.

### Değerlendirme Ölçütleri ve Bilimsel Katkı

Model performansı; eğitim ve doğrulama setleri üzerinden elde edilen doğruluk (Accuracy) ve kayıp (Loss) gibi temel derin öğrenme performans ölçütleriyle değerlendirilmiştir. Bu değerlendirmeler sayesinde sistemin işaret tanıma süreçlerinin başarısı nesnel olarak ortaya konulmuştur.

Çalışmanın bilimsel katkısı, Türk İşaret Diline özgü verilerle eğitilmiş, geleneksel ardışık ağlar (RNN/LSTM) yerine modern Transformer mimarisini kullanan ve sesi de işin içine katan uçtan uca bir sistem geliştirilmesidir. Bu yönleriyle çalışma, hem sosyal hem de teknolojik bir yenilik sunmakta ve işitme engelli bireylerin dijital dünyaya daha etkin şekilde katılımını desteklemektedir.

---

# 2. KAYNAK ARAŞTIRMASI

Bu bölümde, işaret dili tanıma, çeviri ve seslendirme teknolojileri üzerine yapılmış ulusal ve uluslararası araştırmalar incelenmiştir. Literatürde yer alan çalışmalar; kullanılan veri türü (sensör veya kamera tabanlı), yöntem (geleneksel makine öğrenmesi, evrişimli/tekrarlayan derin öğrenme ağları, öz-dikkat (self-attention) tabanlı Transformer mimarileri, çok modlu çeviri vb.) ve amaç (tanıma, metne dönüştürme, seslendirme) kriterlerine göre sınıflandırılmıştır.

Aşağıdaki alt başlıklarda, ilk sensör tabanlı sistemlerden günümüzün iskelet (landmark) tabanlı ve derin öğrenme destekli çok modlu çözümlerine kadar olan gelişim hem kronolojik hem de tematik bir yaklaşımla açıklanmıştır.

## 2.1. İşaret Dili Tanıma Sistemlerine Genel Bakış

İşaret dili tanıma (Sign Language Recognition - SLR) sistemleri, işaret hareketlerinin bilgisayar tarafından algılanarak metin veya ses biçimine dönüştürülmesini amaçlayan çalışmalardır. Bu alandaki ilk araştırmalar genellikle sensör tabanlı eldiven sistemlerine dayanmış; daha sonra görüntü tabanlı yaklaşımlar, derin öğrenme modelleri ve gerçek zamanlı sistemlerle geliştirilmiştir.

2007–2017 yılları arasındaki eldiven tabanlı sistemleri kapsamlı biçimde inceleyerek, sensörlerin hareket yakalama doğruluğunu artırdığını, ancak kullanıcı konforu ve donanım maliyeti açısından sınırlamalar bulunduğunu belirtmiştir (Ahmed ve ark., 2018).

## 2.2. Sensör Tabanlı İşaret Dili Tanıma Sistemleri

Sensör tabanlı sistemler, işaret dili tanıma çalışmalarının ilk kuşağını temsil etmektedir. Bu sistemlerde eldiven veya bileklik içine yerleştirilen ivmeölçer (accelerometer), jiroskop (gyroscope) ve bükülme sensörleri, parmak ve bilek hareketlerini sayısal verilere dönüştürür. Avantajları arasında ışık koşullarından bağımsız çalışma ve yüksek hareket hassasiyeti bulunurken, dezavantajları arasında donanım maliyeti, kullanıcıya bağlı kalma zorunluluğu ve sınırlı doğal etkileşim yer almaktadır.

Çin İşaret Dili için geliştirdikleri çok düğümlü mikro ataletsel ölçüm birimi (IMU) sisteminde, 12 sensör düğümünden elde edilen verileri Kalman filtresi, DCT (Discrete Cosine Transform) ve HMM (Hidden Markov Model) algoritmalarıyla işlemiş ve %95–100 arasında tanıma başarımı elde etmiştir (Tu ve ark., 2015).

Bu çalışma, sensör tabanlı sistemlerin özellikle laboratuvar ortamında yüksek doğruluk sağlayabildiğini, ancak günlük kullanımda kamera tabanlı yaklaşımlara kıyasla daha az pratik olduğunu göstermektedir.

Dolayısıyla sensör sistemleri, bu alandaki gelişimin önemli bir kilometre taşı olmakla birlikte, daha sonraki çalışmalar kamera ve derin öğrenme temelli çözümlere yönelmiştir.

## 2.3. Görsel (Kamera) Tabanlı İşaret Dili Tanıma Sistemleri

Görsel tabanlı işaret dili tanıma sistemleri, sensör tabanlı çözümlere kıyasla daha doğal, temassız ve kullanıcı dostu bir etkileşim sağlamaktadır. Geleneksel olarak doğrudan video piksellerini kullanan evrişimli sinir ağları (CNN) yaygın olarak tercih edilmiş olsa da, son yıllarda ham görüntü yerine insan vücudu, el ve yüz anahtar noktalarını (landmarks) kullanan iskelet tabanlı yaklaşımlar öne çıkmaktadır. Bu tür anahtar nokta çıkarımı için yaygın olarak kullanılan çerçevelerden biri olan MediaPipe, gerçek zamanlı ve yüksek doğruluklu landmark tespiti sağlamaktadır (Lugaresi ve ark., 2019). İskelet verileri, arka plan gürültüsünden ve ışık değişimlerinden bağımsız, hesaplama açısından daha verimli bir temsil sunarak modelin doğrudan hareketin uzamsal ve zamansal dinamiklerine odaklanmasını sağlamaktadır (Pu ve ark., 2024).

Zaman serisi verilerinin modellenmesinde geleneksel tekrarlayan sinir ağlarının (RNN/LSTM) yerini, veriyi paralel işleyebilen ve uzun menzilli bağımlılıkları (long-range dependencies) etkili bir şekilde öğrenebilen Transformer mimarileri almıştır. Bu mimarilerin işaret dili alanındaki ilk başarılı uygulamalarından biri, işaret dilini hem tanıma hem de çeviri görevlerinde uçtan uca ele alan Transformer tabanlı yaklaşımlardır (Camgoz ve ark., 2020). Doğal dil işleme alanında devrim yaratan öz-dikkat (self-attention) mekanizmasına dayalı bu mimariler, işaret dili tanıma görevlerinde de üstün başarı göstermeye başlamıştır. Sadece öz-dikkat mekanizmalarına dayanan bu yapılar, hareketin tüm aşamaları arasındaki karmaşık ilişkileri modelleyerek işaret dilinin hem uzamsal hem de zamansal özelliklerini başarıyla yakalayabilmektedir (Vaswani ve ark., 2017).

Özellikle Türk İşaret Dili (TİD) için geliştirilen geniş ölçekli AUTSL veri seti, bu modern mimarilerin eğitilmesi için önemli bir zemin oluşturmuştur. AUTSL veri seti üzerinde yapılan güncel çalışmalar, iskelet verilerini ve uzamsal-zamansal (spatial-temporal) modelleme tekniklerini birleştirerek yüksek tanıma oranlarına ulaşmıştır. Örneğin, iskelet tabanlı grafik ve dikkat mekanizmalarını kullanan çoklu ipucu (multi-cue) zaman modelleme yaklaşımları veya tekrarlayan grafik sinir ağları ile izole işaretlerin sınıflandırılmasında dikkate değer doğruluk elde edilmiştir. Bu çalışmalar, el ve üst gövde iskelet topolojisinin işaret dili tanımadaki kritik rolünü doğrulamaktadır (Özdemir ve ark., 2023; Mederos ve ark., 2025).

## 2.4. Gerçek Zamanlı (Real-Time) İşaret Dili Tanıma Uygulamaları

GesSpy adlı çalışmada MediaPipe, TensorFlow ve Scikit-Learn kullanarak gerçek zamanlı bir işaret dili tespiti sistemi geliştirmiştir. Sistem, kameradan alınan görüntüleri anında işleyip metin ve ses çıktısı üretmiş, %70–90 doğruluk aralığında performans göstermiştir (Ansari ve ark., 2024).

TD-Net modeli de gerçek zamanlı performansıyla literatürde önemli bir konuma sahiptir. Bu tür yaklaşımlar, işaret dilinin canlı ortamda tanınması için gerekli işlem hızını sağlamaktadır (Nguyen ve ark., 2022).

## 2.5. İşaret Dili Çeviri ve Seslendirme Sistemleri

İşaret dili tanıma sistemlerinin son aşaması, tanınan işaretlerin metne ve ardından konuşmaya dönüştürülmesidir. Bu süreçte hem metinden konuşmaya (Text-to-Speech - TTS) teknolojileri hem de çok modlu çeviri (multimodal translation) yaklaşımları önemli bir rol oynamaktadır.

MB-iSTFT-VITS adlı hafif ve yüksek kaliteli bir metinden konuşmaya sistemi önermiştir. Bu model, Inverse Short-Time Fourier Transform (iSTFT) ve multi-band generation tekniklerini birleştirerek 4,1 kat daha hızlı çalışmış, ancak ses kalitesi açısından orijinal VITS modeliyle eşdeğer performans sergilemiştir. Bu tür hafif TTS yapıları, özellikle gerçek zamanlı işaret dili çeviri projelerinde ses çıktısının hızlı ve doğal biçimde üretilmesi için oldukça uygun bir çözüm sunmaktadır (Kawamura ve ark., 2023).

Öte yandan, Multimodal Neural Machine Translation (MNMT) sistemleri ele alınmış ve metin, görüntü ile ses verilerinin birlikte kullanılmasıyla çeviri doğruluğunun artırılabileceği gösterilmiştir. Araştırmacılar, hiyerarşik füzyon (hierarchical fusion) yönteminin %91,3 ile en yüksek doğruluk oranını sağladığını belirtmişlerdir. Bu yaklaşım, işaret–metin–konuşma bütünleşik çeviri sistemlerinin geliştirilmesi açısından literatürde önemli bir adım olarak değerlendirilmiştir (Nair ve ark., 2023).

## 2.6. Genel Değerlendirme ve Literatürdeki Boşluklar

Literatürdeki araştırmalar incelendiğinde, işaret dili tanıma sistemlerinin sensör tabanlı çözümlerden derin öğrenme tabanlı sistemlere, özellikle de öz-dikkat mekanizmalı Transformer mimarilerine ve iskelet (landmark) tabanlı veri işleme tekniklerine doğru güçlü bir geçiş yaptığı görülmektedir. Bununla birlikte, mevcut çalışmaların büyük bir kısmı sadece "işaretten metne" dönüşüm sağlamış ve genellikle kısıtlı bir kelime dağarcığına odaklanmıştır. Türk İşaret Dili (TİD) literatüründe, geniş ölçekli bir sözlük yapısını (örneğin AUTSL veri setindeki 226 sınıfı) destekleyen, hafif iskelet verilerini (landmarks) analiz eden ve aynı zamanda elde edilen metni gerçek zamanlı olarak sese (TTS) dönüştüren bütünleşik (end-to-end) sistemlerin eksikliği dikkat çekmektedir. Bu bağlamda projemiz; bilgisayarlı görü ile elde edilmiş optimize iskelet verilerini yalnızca kodlayıcı (Encoder-only) tabanlı bir Transformer modeliyle işleyerek TİD'e ait 226 farklı işareti tanıyan ve bunu anında konuşmaya dönüştüren yapısıyla literatürdeki bu önemli boşluğu doldurmayı hedeflemektedir.

---

# 3. MATERYAL VE YÖNTEM

Bu bölüm, Türk İşaret Dili’ni (TİD) yapay zekâ teknikleri aracılığıyla metin ve konuşmaya dönüştüren sistemin geliştirilme sürecinde kullanılan bilimsel metodolojiyi, teknik araçları ve uygulama adımlarını detaylandırmaktadır. Sistemin mimarisi, görüntü tabanlı verilerin iskeletsel koordinatlara dönüştürülmesi ve bu zamansal verilerin derin öğrenme modelleri ile anlamlandırılması prensibine dayanmaktadır. Geliştirilen uçtan uca (end-to-end) boru hattı (pipeline); veri toplama, öznitelik çıkarımı, Transformer tabanlı zamansal modelleme ve Metinden Konuşmaya Çeviri (Text-to-Speech - TTS) aşamalarından oluşmaktadır. Bu bölümde, sistemin gerçek zamanlı çalışma performansını ve tanıma doğruluğunu optimize etmek için tercih edilen algoritmik yaklaşımlar ve mühendislik çözümleri akademik bir çerçevede sunulmaktadır. Bu doğrultuda, önerilen sistemin temel bileşenlerini ve veri akışını gösteren genel mimari Şekil 1'de şematize edilmiştir.

_Şekil 1: Sistemin Genel İş Akış Şeması (İşaret Dili - Landmark - Transformer - TTS)._

## 3.1. Çalışma Ortamı ve Donanım/Yazılım Araçları

Sistemin geliştirilmesi ve yüksek boyutlu iskeletsel verilerin işlenmesi sürecinde, hesaplama verimliliğini artırmak ve eğitim sürelerini optimize etmek amacıyla modern donanım ve yazılım altyapılarından faydalanılmıştır. Çalışma ortamına ilişkin teknik detaylar aşağıda maddeler hâlinde sunulmuştur:

### 3.1.1. Yazılım araçları

- **Python Programlama Dili:** Veri işleme, modelleme ve entegrasyon süreçlerinde sağladığı geniş kütüphane desteği ve esnek yapısı nedeniyle ana programlama dili olarak tercih edilmiştir.
- **OpenCV (Açık Kaynak Bilgisayarlı Görü Kütüphanesi):** Video akışlarının yakalanması, karelerin (frames) işlenmesi ve görüntü ön işleme adımlarının gerçekleştirilmesi amacıyla kullanılmıştır.
- **MediaPipe Holistic:** Vücut pozisyonu (pose), eller ve yüz anahtar noktalarının (landmarks) eş zamanlı ve yüksek hassasiyetle çıkarılması için temel bilgisayarlı görü aracı olarak entegre edilmiştir.
- **Derin Öğrenme Çatıları (TensorFlow / Keras):** Önerilen Transformer (Encoder-only) mimarisinin tasarlanması, hiper-parametre optimizasyonu ve modelin eğitilmesi süreçlerinde kullanılmıştır.
- **Metinden Konuşmaya (TTS) Motoru:** Elde edilen metin tabanlı çıktıların doğal bir ses formuna dönüştürülerek kullanıcıya iletilmesini sağlamak amacıyla sisteme dahil edilmiştir.

### 3.1.2. Donanım araçları ve hesaplama platformu

- **Görüntü Yakalama Birimi (Camera):** İşaret dili hareketlerini akıcı ve detaylı bir şekilde yakalayabilmek amacıyla saniyede 30 kare (30 FPS) hızında kayıt yapabilen yüksek çözünürlüklü dijital kamera kullanılmıştır. Bu kare hızı, hızlı el hareketlerinin takibi ve modelin zaman serisi analizindeki başarısı için kritik bir öneme sahiptir.
- **Bulut Tabanlı Hesaplama (Google Colab):** Modelin eğitim aşamasında ihtiyaç duyulan yüksek işlem gücü, Google Colab platformu üzerinden sağlanmıştır. Özellikle derin öğrenme modelinin eğitim sürecini hızlandırmak amacıyla Google Colab bünyesindeki yüksek performanslı Grafik İşlem Birimleri (GPU - Tesla T4/L4) kullanılarak hesaplama süresi minimuma indirilmiş ve donanım kısıtlamaları aşılamıştır.

## 3.2. Veri Seti Tanımı ve Veri Yapısı

Bu çalışmada önerilen sistemin eğitimi ve test edilmesi süreçlerinde, Ankara Üniversitesi tarafından geliştirilen ve literatürde geniş ölçekli bir referans kabul edilen AUTSL (Ankara University Turkish Sign Language) veri seti kullanılmıştır. AUTSL, Türk İşaret Dili (TİD) üzerine yapılan bilimsel araştırmalar için sunulmuş, çok modlu (multi-modal) ve izole işaret videolarından oluşan kapsamlı bir veri kümesidir.

### 3.2.1. Veri setinin kapsamı ve istatistiksel özellikleri

AUTSL veri seti, günlük konuşma dilinde en sık kullanılan 226 farklı işareti içermektedir. Bu işaretler, veri çeşitliliğini artırmak amacıyla 43 farklı işaret dili kullanıcısı (signer) tarafından icra edilmiştir. Veri setinde toplamda 38.336 adet izole işaret videosu bulunmaktadır. Her bir video örneği, işaretin başlangıcından bitişine kadar olan süreci kapsayan dinamik bir hareket dizisidir.

### 3.2.2. Teknik spesifikasyonlar ve çevresel zorluklar

Veriler, Microsoft Kinect v2 sensörü kullanılarak kaydedilmiş olup orijinal yapısında RGB (renkli görüntü), derinlik (depth) ve sensör tabanlı iskelet (skeleton) verilerini barındırmaktadır. Görüntüler standart olarak 512×512 piksel çözünürlükte sunulmuştur. Veri setinin en önemli ayırt edici özelliği, gerçek dünya senaryolarını simüle eden çevresel zorlukları barındırmasıdır:

- **Arka Plan Çeşitliliği:** Videolar 20 farklı kapalı ve açık alan arka planında kaydedilmiştir. Bu durum, modelin karmaşık arka planlar karşısındaki dayanıklılığını (robustness) ölçmektedir.
- **Işık ve Pozisyon Değişimleri:** Yüksek ışık değişimleri, yansımalar, işaretçilerin farklı kıyafetleri, oturma veya ayakta durma gibi postür farklılıkları veri setinin varyansını artırmaktadır. Bahsedilen bu çevresel çeşitliliği, farklı ışıklandırma koşullarını ve mekânsal değişiklikleri gösteren örnek veri seti görüntüleri Şekil 2’de sunulmuştur.

_Şekil 2: Veri Setindeki Farklı Arka Plan, Işıklandırma ve Konum Çeşitliliğini Gösteren Örnek Görüntüler._

### 3.2.3. Proje kapsamında veri kullanımı ve landmark çıkarımı

AUTSL veri seti orijinalinde çok modlu veriler sunmasına rağmen, bu projede önerilen "temassız ve hafif mimari" hedefi doğrultusunda yalnızca RGB modalitesi temel girdi olarak benimsenmiştir. Bu kapsamda;

- Veri setindeki RGB videolar karelerine (frames) ayrılmıştır.
- Her bir kareden, derinlik sensörüne ihtiyaç duymadan, MediaPipe Holistic kütüphanesi aracılığıyla vücut, el ve yüz anahtar noktaları (landmarks) çıkarılmıştır.
- Böylece, ham görüntü verisi; zaman serisi analizine uygun, gürültüden arındırılmış ve koordinat bazlı bir iskeletsel temsil yapısına dönüştürülmüştür.

### 3.2.4. Değerlendirme protokolü

Modellerin başarısını objektif bir şekilde değerlendirmek için veri setinin sunduğu "Kullanıcıdan Bağımsız" (User-Independent) test protokolü uygulanmıştır. Bu doğrultuda, eğitim ve doğrulama setlerinde yer almayan 6 farklı işaretçiden elde edilen veriler ayrı bir test seti olarak ayrılmıştır. Bu yaklaşım, sistemin daha önce hiç görmediği bireylerin işaretlerini tanıma yeteneğini ölçerek akademik geçerliliği sağlamaktadır (Sincan ve Keles, 2020).

## 3.3. MediaPipe Holistic ile İskeletsel Öznitelik Çıkarımı

Sistemin temelini oluşturan öznitelik çıkarımı aşamasında, ham video karelerini (pixel data) anlamlı geometrik temsillere dönüştürmek amacıyla MediaPipe Holistic kütüphanesi kullanılmıştır. Bu yaklaşım, görüntüdeki karmaşık arka plan ve ışık gürültülerini eleyerek, modelin yalnızca işaret dilinin anlamsal özünü taşıyan hareketlere odaklanmasını sağlamaktadır.

### 3.3.1. Optimize edilmiş landmark seçimi ve veri sadeleştirme

İşaret dili tanıma görevinde, vücudun tüm noktalarının işlenmesi model üzerinde gereksiz bir hesaplama yükü (computational overhead) oluşturabilmektedir. Bu projede, modelin hem doğruluğunu korumak hem de işlem hızını optimize etmek amacıyla stratejik bir "nokta seyreltme" yöntemi benimsenmiştir. MediaPipe Holistic modeli tarafından sağlanan 543 koordinatın tamamı yerine, işaret dilinin anlamsal özelliklerini taşıyan toplam 85 temel anahtar nokta (landmark) seçilmiştir. Bu noktalar; üst gövde pozisyonunu, sağ ve sol elin tüm eklem noktalarını (21x2 = 42 nokta) ve yüz mimiklerini (dudak hareketlerini) temsil eden kritik koordinatlardan oluşmaktadır. Her bir nokta 3 boyutlu uzayda (x, y, z) ifade edildiği için, her bir video karesi (frame) modelin işleyebileceği 255 boyutlu (85x3) bir öznitelik vektörüne dönüştürülmüştür. Bu yapısal sadeleştirme, modelin eğitim sürecinde aşırı öğrenmeyi (overfitting) engellemekte ve gerçek zamanlı çıkarım (real-time inference) hızını büyük ölçüde artırmaktadır.

_Şekil 3: MediaPipe Tarafından Sağlanan Tüm Ham İşaret Noktaları (Sol) ile Model İçin Seçilen Optimize Edilmiş Noktaların (Sağ) Karşılaştırılması._

### 3.3.2. Koordinat sistemi ve öznitelik vektörü oluşturma

Çıkarılan her bir anahtar nokta, üç boyutlu uzayda (x, y, z) koordinatları ile temsil edilmektedir. Burada 'x' ve 'y' değerleri görüntünün genişlik ve yüksekliğine göre normalize edilmiş değerleri, 'z' değeri ise noktanın kameraya olan göreceli derinliğini ifade etmektedir.

Her bir video karesi için elde edilen bu alt öznitelik grupları (üst gövde, eller ve seçili yüz noktaları), ardışık olarak birleştirilerek (concatenation) tek bir Öznitelik Vektörü (Feature Vector) hâline getirilmektedir. Sonuç olarak, her video örneği [T×N] boyutunda bir matris yapısına dönüştürülmektedir; burada T zaman içindeki kare sayısını, N ise her karedeki toplam koordinat sayısını temsil etmektedir. Bu yapılandırılmış veri seti, hareketin hem uzamsal (spatial) hem de zamansal (temporal) bilgisini Transformer mimarisine aktarmak için hazır hâle getirilmiş olur.

## 3.4. Veri Ön İşleme ve Normalizasyon

MediaPipe Holistic kütüphanesi aracılığıyla çıkarılan ham iskeletsel veriler (landmarks), işaretçinin kameraya olan mesafesine, görüntü içindeki konumuna ve fiziksel vücut yapısına bağlı olarak yüksek varyans içermektedir. Derin öğrenme modelinin bu değişkenlerden bağımsız olarak yalnızca hareketin karakteristiğine odaklanabilmesi ve eğitim sürecinin kararlılığını artırmak amacıyla kapsamlı bir ön işleme ve normalizasyon süreci uygulanmıştır.

### 3.4.1. Uzamsal normalizasyon

Modelin, işaretçinin görüntü çerçevesindeki konumundan (merkezde, sağda veya solda olması) etkilenmemesi için koordinat bazlı bir dengeleme yöntemi izlenmiştir:

- **Merkezleme (Centering):** Her bir karedeki tüm anahtar noktalar, vücudun üst kısmındaki sabit bir referans noktasına (örneğin omuz merkez noktası) göre yeniden konumlandırılmıştır. Bu işlem, referans noktasının koordinatlarının diğer tüm noktalardan çıkarılmasıyla gerçekleştirilerek verinin orijin noktasına taşınması sağlanmıştır.
- **Ölçeklendirme (Scaling):** İşaretçilerin farklı vücut boyutlarına sahip olmasından kaynaklanan sapmaları gidermek için, koordinat değerleri iki omuz arasındaki mesafe gibi anatomik bir birime bölünerek normalize edilmiştir. Böylece sistem, "kişiden bağımsız" (person-independent) bir çalışma yapısına kavuşturulmuştur. Uygulanan bu merkezleme ve ölçeklendirme işlemlerinin, ham veriyi nasıl standart ve hizalanmış bir yapıya dönüştürdüğü Şekil 4’te aşamalı olarak gösterilmektedir.

_Şekil 4: Ham İşaret Noktalarından (Sol) Vücut Merkezli Normalizasyon ile Standartlaştırılmış Veriye (Sağ) Geçiş Süreci._

### 3.4.2. Zamansal hizalama: padding ve masking

İşaret dili hareketlerinin icra sürelerinin (kare sayılarının) farklılık göstermesi nedeniyle, zamansal girdileri Transformer modelinin mimarisine uyumlu hale getirmek ve canlı akış (streaming) kararlılığını sağlamak için iki aşamalı bir strateji izlenmiştir:

- **Eğitim Aşaması (Zamansal Yeniden Boyutlandırma):** Çevrimdışı (offline) eğitim sürecinde, AUTSL veri setindeki farklı uzunluktaki videoları standartlaştırmak amacıyla "Çift Doğrusal Ara Değerleme" (Bilinear Interpolation) tekniği kullanılmıştır. Bu yöntemle tüm video dizileri, hareketin akıcılığı korunarak sabit 80 kare (frame) uzunluğuna getirilmiştir.
- **Çıkarım Aşaması (İlksel Kare Tekrarı ve Kayan Pencere):** Gerçek zamanlı (real-time) kullanımda, sistemin giriş yığını (batch) beklememesi için 80 kare kapasiteli bir "Kayan Pencere" (Sliding Window) mekanizması geliştirilmiştir. Geleneksel literatürde eksik kareler için kullanılan "sıfır dolgulama" (zero-padding) yöntemi, Transformer modelinin öz-dikkat (self-attention) mekanizmasında ani sinyal değişimlerine ve kararsız tahminlere (flickering) neden olabilmektedir. Bu sorunu aşmak amacıyla, buffer (geçici bellek) henüz dolmadan yapılan tahminlerde "İlksel Kare Tekrarı" (First-Frame Replication) tekniği uygulanmıştır. Bu teknikte, eksik kareler anlamsız sıfırlar yerine dizinin ilk karesiyle doldurularak, işaretçinin harekete başlamadan önceki "hareketsiz duruş" (resting pose) anı simüle edilmiştir. Bu yaklaşım, modelin dikkat dağılımını stabilize ederek canlı kamera akışında daha tutarlı bir sınıflandırma performansı sunmasını sağlamıştır.

### 3.4.3. Veri temizleme ve kalite kontrol

İskelet verilerinin kalitesini artırmak için çıkarım sürecinde düşük güven skoruna (confidence score) sahip olan veya hatalı tespit edilen kareler tespit edilmiştir. MediaPipe tarafından sağlanan görünürlük skorları kullanılarak, anahtar noktaların eksik veya hatalı olduğu kareler elenmiş veya doğrusal interpolasyon yöntemleriyle düzeltilmiştir. Bu temizlik aşaması, modelin gürültülü verilerle eğitilmesini önleyerek tanımanın doğruluğunu doğrudan olumlu yönde etkilemiştir.

## 3.5. Transformer Tabanlı Zamansal Modelleme

İşaret dili, el ve vücut hareketlerinin zaman içindeki değişimine ve bu değişimlerin bir bütün olarak ifade ettiği anlama dayalı dinamik bir dildir. Bu çalışmada, iskeletsel anahtar noktaların zamansal dizilimini yüksek doğrulukla sınıflandırmak amacıyla modern Transformer (Encoder-only) mimarisi kullanılmıştır. Geleneksel ardışık modellerin (LSTM/RNN) aksine Transformer, veriyi paralel işleme yeteneği ve "Çok Başlı Öz-Dikkat" (Multi-Head Self-Attention) mekanizması ile hareketin tüm aşamaları arasındaki karmaşık ilişkileri aynı anda değerlendirebilmektedir.

### 3.5.1. Kodlayıcı Temelli Mimari ve Özellik Projeksiyonu

Sistemin mimarisi, işaret tanıma bir sınıflandırma görevi olduğu için yalnızca Transformer bloklarının "Kodlayıcı" kısmından oluşacak şekilde tasarlanmıştır. Girdi olarak alınan [80, 255] boyutundaki normalize edilmiş landmark matrisi, ilk aşamada bir Doğrusal Projeksiyon (Linear Projection) katmanından geçirilerek 256 boyutlu özellik uzayına (embed_dim = 256) aktarılmıştır. Bu projeksiyon, modelin yüksek boyutlu koordinat verilerini daha anlamlı vektör temsillerine dönüştürmesini sağlamaktadır.

### 3.5.2. Öğrenilebilir Konumsal Gömme (Positional Embedding)

Transformer mimarisi, veriyi paralel olarak işlediği için dizideki karelerin zamansal sırasına ilişkin yerleşik bir bilgiye sahip değildir. İşaret dilinde hareketin sırası (örneğin elin yukarıdan aşağıya inmesi) anlamı tamamen değiştirdiği için, özellik vektörlerine Öğrenilebilir Konumsal Gömme (Learned Positional Embedding) eklenmiştir. Bu katman, her bir kareye dizideki konumuna özgü bir vektör ekleyerek modelin hareketin kronolojik akışını ve hiyerarşik yapısını kavramasını sağlamıştır.

### 3.5.3. Çok Başlı Öz-Dikkat (Multi-Head Self-Attention) Mekanizması

Modelin çekirdeğini oluşturan 3 ardışık Transformer bloğu, 8 başlı (8-heads) öz-dikkat mekanizması ile yapılandırılmıştır. Bu mekanizma, her bir video karesinin dizideki diğer tüm karelerle olan anlamsal korelasyonunu hesaplayarak; işaretin başlangıcındaki hazırlık hareketi ile sonundaki bitiş hareketi arasında doğrudan bağlantı kurabilmektedir. Bu sayede model, "Uzun Menzilli Bağımlılıkları" (Long-range dependencies) bilgi kaybı yaşamadan korumakta ve hareketin en kritik anlarına (örneğin elin tam şekil aldığı an) daha yüksek ağırlık vermektedir.

### 3.5.4. Sınıflandırma ve Optimizasyon Stratejisi

Transformer bloklarından elde edilen bağlamsal özellikler, Global Ortalama Havuzlama (Global Average Pooling) katmanı ile boyut küçültme işlemine tabi tutulmuş ve ardından 226 sınıf için Softmax aktivasyon fonksiyonuna iletilmiştir. Modelin eğitiminde; aşırı öğrenmeyi (overfitting) engellemek için Label Smoothing (0.1) ve AdamW optimizasyon algoritması kullanılmıştır. Ayrıca, öğrenme oranını dinamik olarak ayarlayan "Isınma ile Kosinüs Azalması" (Cosine Decay with Warmup) stratejisi ile modelin yerel minimum noktalarına daha istikrarlı bir şekilde yakınsaması sağlanmıştır.

Yukarıdaki bölümlerde (3.5.1 - 3.5.4) detaylandırılan tüm bileşenlerin entegrasyonu, önerilen sistemin bütüncül yapısını oluşturmaktadır. İşaret dili videolarından çıkarılan normalize edilmiş landmark dizilerinin, giriş projeksiyonundan başlayarak, pozisyonel kodlama ile zaman damgası alması, ardından Transformer kodlayıcı bloklarında öz-dikkat mekanizması ile işlenmesi ve son olarak sınıflandırma katmanında bir olasılık dağılımına dönüştürülmesi süreci görselleştirilmiştir. Bu uçtan uca (end-to-end) veri akışı ve önerilen "Kodlayıcı Temelli" mimarinin şematik gösterimi Şekil 5’te sunulmaktadır.

_Şekil 5: Önerilen Transformer (Encoder-only) Model Mimarisi ve Veri Akış Diyagramı. (Yapay zeka desteği ile tasarlanmıştır.)_

## 3.6. Metin ve Konuşma Dönüştürme Süreci

İşaret dili tanıma sisteminin son aşaması, Transformer modelinden elde edilen yüksek boyutlu özniteliklerin ve sınıflandırma tahminlerinin, kullanıcı için anlamlı, okunabilir ve işitsel bir forma dönüştürülmesini kapsamaktadır. Bu süreç, gerçek zamanlı iletişim (real-time communication) dinamiklerine uygun olarak tasarlanmıştır.

### 3.6.1. Sınıflandırma sonuçlarının metinleştirilmesi ve çıkarım istikrarı

Transformer modelinin son katmanı olan Softmax fonksiyonundan gelen olasılık dağılımları, önceden tanımlanmış olan sözlük (mapping) yapısı ile eşleştirilmektedir. Gerçek zamanlı kamera akışında anlık hatalı tahminleri (false positives) önlemek amacıyla sisteme iki aşamalı bir güvenlik mekanizması entegre edilmiştir. İlk olarak, bir "Hareket Etkinliği Filtresi" (Sign Activity Filter) aracılığıyla kullanıcının o an gerçekten bir işaret dili hareketi yapıp yapmadığı (ellerin pozisyonu ve hareket hızına bakılarak) kontrol edilmektedir. İkinci olarak, arka arkaya gelen karelerdeki tahminlerin kararlılığını ölçen bir "Oylama Mekanizması" (Voting Mechanism) kullanılarak, yalnızca ardışık olarak tutarlı bir şekilde tahmin edilen işaretler nihai metin çıktısı olarak kabul edilmektedir. En yüksek ve kararlı olasılık puanına sahip sınıf etiketi, AUTSL veri setindeki 226 farklı işaret arasından Türkçe kelime (gloss) olarak dijital ortama aktarılmaktadır.

### 3.6.2. Metinden konuşmaya çeviri (Piper TTS) entegrasyonu

Sistemin kapsayıcılığını artırmak ve işitme engelli bireyler ile toplum arasındaki iletişimi daha doğal bir boyuta taşımak amacıyla, elde edilen metin çıktıları modern ve yerel olarak çalışabilen Piper TTS (Text-to-Speech) motoruna entegre edilmiştir. Bu modül, internet bağlantısına ihtiyaç duymadan cihaz üzerinde (on-device) hızlı ve yüksek kaliteli ses sentezi yapabilmektedir. Sistem, hedef dil olan Türkçenin fonetik yapısına tam uyumlu olacak şekilde yapılandırılmış olup, sentezlenen sesin robotik tınıdan uzak, doğal insan konuşma ritmine (prosody) en yakın seviyede olması hedeflenmiştir.

### 3.6.3. Sistem çıktısının bütünleştirilmesi

Uçtan uca mimarinin son halkası olarak, Transformer modelinin tanıdığı ve oylama mekanizmasından başarıyla geçen her bir işaret, Piper TTS aracılığıyla milisaniyeler içerisinde ses dalgasına (WAV/Audio akışı) dönüştürülerek kullanıcıya iletilmektedir. Bu hızlı ve kesintisiz dönüşüm süreci, sistemin "gerçek zamanlı iletişim köprüsü" olma hedefini tamamlamaktadır. Böylece, işaret diliyle ifade edilen bir kavram, karşıdaki kişi tarafından hem yazılı hem de anında sözlü olarak algılanabilir hâle gelmektedir.

_Şekil 6: Model Çıktısından Ses Sentezine Geçiş Şeması._

---

# 4. ARAŞTIRMA SONUÇLARI VE TARTIŞMA

Bu bölümde, projenin temelini oluşturan Transformer (Encoder-only) modelinin eğitim süreci, test seti üzerindeki performansı ve gerçek zamanlı sistem entegrasyonuna dair elde edilen bulgular sunulmaktadır. Sadece 85 optimize edilmiş iskeletsel anahtar nokta (landmark) kullanılarak geliştirilen bu mimarinin, geniş kelime dağarcığına (226 sınıf) sahip AUTSL veri setindeki başarısı nesnel metriklerle analiz edilmiş ve literatürdeki mevcut çalışmalarla karşılaştırılmıştır.

_Şekil 7: İşaret Dilinden Konuşmaya Dönüşüm Sisteminin Uçtan Uca Çalışma Mimarisi ve Aşamalı Veri Akış Şeması (Yapay zeka desteği ile tasarlanmıştır)._

## 4.1. Eğitim Süreci ve Hiperparametre Optimizasyonu

Modelin eğitim süreci, Kaggle bulut platformu üzerinde GPU (Tesla P100) donanımı kullanılarak toplam 81.3 dakika sürmüştür. 7.7 milyon öğrenilebilir parametreye sahip olan ağ, 28.130 eğitim örneği üzerinden 200 epok (epoch) boyunca eğitilmiştir. Eğitim sürecinde, modelin karmaşık işaretleri ezberlemesini (overfitting) engellemek ve genelleme yeteneğini artırmak için "Etiket Yumuşatma" (Label Smoothing = 0.1) ve "Ağırlık Azaltma" (Weight Decay) teknikleri uygulanmıştır.

Ayrıca, öğrenme oranının (learning rate) optimizasyonunda klasik sabit oranlar yerine "Isınma ile Kosinüs Azalması" (Cosine Decay with Warmup) stratejisi benimsenmiştir. Bu strateji sayesinde model, ilk 15 epok boyunca kademeli olarak hızlanmış ve ardından yavaşlayarak yerel minimum noktalarına istikrarlı bir şekilde oturmuştur. Şekil 8’de, modelin eğitim ve doğrulama setleri üzerindeki doğruluk (accuracy), kayıp (loss) ve öğrenme oranı (learning rate) eğrileri sunulmuştur. Grafikler incelendiğinde, eğitim ve doğrulama eğrilerinin birbirine paralel ilerlediği ve aşırı öğrenme (overfitting) probleminin başarıyla aşıldığı görülmektedir. Model, en yüksek doğrulama doğruluğuna (%89.99) 172. epokta ulaşmıştır.

_Şekil 8: Transformer modelinin 200 epokluk eğitim sürecinde elde edilen eğitim (mavi) ve doğrulama (kırmızı) setlerine ait doğruluk (sol) ve kayıp (sağ) değişim eğrileri._

## 4.2. Modelin Test Seti Üzerindeki Başarısı

Eğitim sürecinin ardından, modelin daha önce hiç görmediği kullanıcılardan oluşan 3.739 örnekli test seti üzerinde nihai bir değerlendirme yapılmıştır. Test sonuçları, iskelet tabanlı Transformer mimarisinin 226 farklı işareti ayırt etmedeki üstün yeteneğini ortaya koymuştur. Yapılan nesnel ölçümlerde, modelin genel test doğruluğu (accuracy) %90.96 ve test kaybı (loss) 1.4497 olarak kaydedilmiştir. Sınıflandırma performansının daha kapsamlı bir şekilde incelenmesi amacıyla hesaplanan makro ortalama metriklerinde ise; kesinlik (precision) 0.9156, duyarlılık (recall) 0.9101 ve F1-skoru 0.9088 olarak elde edilmiştir.

Elde edilen bu %90.96'lık yüksek genel doğruluk oranı, modelin izole işaretleri oldukça isabetli bir şekilde metne dönüştürebildiğini kanıtlamaktadır. İşaret dilinin doğası gereği bazı hareketlerin başlangıç ve bitiş pozisyonlarının birbirine çok benzemesi modelleme açısından zorlu bir problem teşkil etse de, model %98.26 gibi olağanüstü bir "İlk-5 Doğruluğu" (Top-5 Accuracy) oranına ulaşmayı başarmıştır. Bu yüksek oran, sistemin en yüksek olasılık atadığı ilk 5 tahmin içinde doğru hedefin bulunma ihtimalinin neredeyse kesin olduğunu göstererek projenin pratik kullanım potansiyelini güçlü bir şekilde pekiştirmektedir.

## 4.3. Sınıf Bazlı Performans ve Karışıklık Matrisi Analizi

Sınıflandırma raporu (Classification Report) ve normalize edilmiş karışıklık matrisi (Confusion Matrix) incelendiğinde, modelin 226 sınıfın büyük çoğunluğunda %90 ile %100 arasında değişen mükemmele yakın doğruluk oranları yakaladığı tespit edilmiştir. Ancak az sayıda da olsa bazı işaret sınıflarında modelin zorlandığı görülmüştür. Örneğin, 144. sınıf (%47.06) ve 6. sınıf (%52.94) en düşük tanıma performansına sahip işaretler olmuştur.

Bu durumun temel teknik ve yapısal nedenleri bulunmaktadır. Seçilen 85 adet anahtar nokta (landmark) vücut, el ve yüz pozisyonlarını başarıyla temsil etse de; bazı Türk İşaret Dili kelimeleri yapısal olarak sadece çok ince parmak boğum hareketlerine veya elin göğüs üzerindeki minimal sürtünme yönlerine dayanmaktadır. İskelet verilerinin doğrudan görsel piksel verisine (RGB) kıyasla doku ve derinlik algısında yaşadığı bu minimal kayıp, görsel (görünüm) olarak birbirine çok benzeyen sınıflar arası (inter-class similarity) karışıklıklara yol açabilmektedir. Buna rağmen makro F1-skorunun 0.9088 seviyesinde kalması, sistemin genel sınıflar arası dengeyi üst düzeyde koruduğunu göstermektedir.

## 4.4. Gerçek Zamanlı Çıkarım ve Entegrasyon Performansı

Akademik başarının yanı sıra, modelin pratik uygulamadaki başarısı da gerçek zamanlı çıkarım (real-time inference) testleri ile ölçülmüştür. Sistemin RGB video piksellerini işlemek yerine kayan pencere (sliding window) mekanizması ile yalnızca boyutları küçültülmüş bir vektör matrisini [1, 80, 255] işlemesi, modelin tahmin süresini milisaniyeler seviyesine indirmiştir.

"Hareket Etkinliği Filtresi" ve "Oylama Mekanizması" ile entegre çalışan sistem, canlı kamera akışında hatalı tetiklenmeleri (false positives) büyük ölçüde filtrelemiştir. Elde edilen tahminlerin cihaz üzerinde çalışan (on-device) Piper TTS motoruyla anında sentezlenmesi sırasında herhangi bir darboğaz (bottleneck) yaşanmamış, sistem "işaretten sese" dönüşüm hedefini akıcı ve kesintisiz bir biçimde yerine getirmiştir.

## 4.5. Literatür ile Karşılaştırma ve Tartışma

Bu çalışmada elde edilen %90.96'lık doğruluk oranı, AUTSL veri seti üzerinde yapılan güncel araştırmalarla karşılaştırıldığında oldukça rekabetçi ve yenilikçidir. Veri setinin orijinal sunumunda yer alan (Sincan ve Keles, 2020) geleneksel CNN+RNN (ResNet, LSTM) tabanlı temel yöntemler (baseline), genellikle yüksek hesaplama gücü gerektiren RGB görüntüleri kullanarak %60-70 bandında doğruluklar elde etmişlerdir. Sonraki çalışmalarda RGB ve derinlik verilerinin birleştirilmesiyle performanslar artırılmış olsa da, sistem gereksinimleri katlanarak büyümüştür.

Bizim geliştirdiğimiz Transformer (Encoder-only) mimarisi ise, ağır görsel veriler yerine sadece MediaPipe ile çıkarılmış optimize iskelet verilerini (landmarks) kullanmasına rağmen %90'ın üzerinde bir doğruluğa ulaşmıştır. Bu durum, öz-dikkat (self-attention) mekanizmalarının hareketin uzamsal-zamansal (spatial-temporal) ilişkilerini kavramada, geleneksel sıralı modellere (LSTM/GRU) kıyasla çok daha verimli olduğunu kanıtlamaktadır. Sonuç olarak önerilen sistem, hem düşük donanım kaynaklarıyla yüksek doğruluk sunması hem de yerel seslendirme (TTS) yeteneğiyle uçtan uca çalışabilmesi bakımından literatüre yenilikçi bir katkı sağlamaktadır.

---

# 5. SONUÇLAR VE ÖNERİLER

Bu bölümde, projenin tasarım, geliştirme ve test aşamalarından elde edilen genel sonuçlar özetlenmekte ve sistemin mevcut durumuna ilişkin değerlendirmeler sunulmaktadır. Ayrıca, projenin tam olarak tamamlanması, nihai bir ürüne dönüştürülmesi ve gelecekteki akademik çalışmalara ışık tutması amacıyla planlanan iyileştirmeler ile teknik öneriler detaylandırılmaktadır.

## 5.1 Sonuçlar

Bu çalışma kapsamında, Türk İşaret Dili (TİD) kelimelerini gerçek zamanlı olarak tanıyarak metne ve sese dönüştüren yapay zekâ tabanlı uçtan uca bir sistem başarıyla prototiplenmiştir. Literatürdeki geleneksel ağır RGB piksel tabanlı ardışık modeller (CNN-RNN) yerine, MediaPipe ile çıkarılan optimize iskelet (landmark) verilerini işleyen hafif ve etkili bir Transformer (Encoder-only) mimarisi tasarlanmıştır.

Sistem, AUTSL veri setinde yer alan günlük yaşama ait 226 farklı işaret sınıfı üzerinde eğitilmiş ve %90.96 gibi oldukça yüksek bir genel test doğruluğu elde etmiştir. İlk-5 (Top-5) doğruluk oranının %98.26'ya ulaşması, modelin işaretleri anlamsal olarak çok dar bir çerçevede doğru tahmin edebildiğini kanıtlamıştır. Yalnızca 85 adet anahtar noktanın kullanılması, hesaplama yükünü ciddi oranda hafifleterek modelin düşük gecikme (low-latency) ile anlık çalışmasını sağlamıştır. Geliştirilen "oylama" (voting) ve "hareket etkinlik" filtreleri sayesinde canlı kamera akışındaki kararlılık maksimize edilmiş; entegre edilen Piper TTS motoru ile işaretten sese dönüşüm süreci cihaz üzerinde (on-device) başarıyla gerçekleştirilmiştir. Bu bulgular, geliştirilen mimarinin TİD tanıma görevlerinde hem akademik hem de pratik bir çözüm olarak güçlü bir potansiyele sahip olduğunu göstermektedir.

## 5.2 Öneriler

Proje mevcut aşamada arka plan algoritmaları, model eğitimi ve gerçek zamanlı çıkarım (inference) motoru açısından başarılı ve istikrarlı sonuçlar vermiş olsa da, sistemin nihai bir ürüne dönüşmesi için geliştirme süreci aktif olarak devam etmektedir. Elde edilen bulgular ve mevcut proje planı doğrultusunda aşağıdaki öneriler ve gelecek çalışmalar belirlenmiştir.

### 5.2.1. Kullanıcı arayüzü geliştirilmesi ve entegrasyonu

Sistemin son kullanıcılar (işitme engelli bireyler ve iletişim kurdukları kişiler) tarafından teknik bilgiye ihtiyaç duyulmadan, kolayca kullanılabilmesi için modern bir grafiksel kullanıcı arayüzü (GUI) tasarlanacaktır. Bu arayüz; canlı kamera akışını, tahmin edilen metni, sistemin anlık durumunu ve seslendirme seçeneklerini tek bir ekranda kullanıcı dostu bir deneyimle sunacaktır.

### 5.2.2. Model optimizasyonu ve performans iyileştirme

Model mimarisi yüksek bir başarı yakalamış olsa da, nihai üründe en iyi sonuca ulaşmak amacıyla hiperparametre optimizasyon çalışmalarına devam edilecektir. Özellikle %90.96'lık test doğruluğunu daha da yukarı taşımak için veri artırma (data augmentation) teknikleri genişletilecek ve kayan pencere (sliding window) mekanizmasının anlık tepki süresi optimize edilecektir.

### 5.2.3. Zorlu sınıflar için çoklu modalite yaklaşımı

Sınıflandırma raporunda tespit edilen ve tanınma oranı nispeten düşük olan (görsel olarak birbirine çok benzeyen) işaretlerin ayrımını kolaylaştırmak için, el şekillerini daha detaylı analiz edecek ek derinlik (depth) verilerinin veya yüz mimiklerini daha hassas ağırlıklandıran dikkat (attention) mekanizmalarının modele dahil edilmesi akademik bir öneri olarak değerlendirilmektedir.

### 5.2.4. Sürekli işaret dili çevirisine geçiş

Mevcut sistem "izole" (tekli) kelimeleri yüksek doğrulukla tanımaktadır. Gelecekteki çalışmalarda, ardışık yapılan işaretleri doğal bir cümle bütünlüğünde çevirebilen ve Türkçenin gramer yapısını anlayan Doğal Dil İşleme (NLP) destekli modellere geçiş yapılması, sistemin kapsamını büyük ölçüde artıracaktır.

---

# 6. KAYNAKLAR

- Ahmed, M. A., Zaidan, B. B., Zaidan, A. A., Salih, M. M., & Lakulu, M. M. B. (2018). A review on systems-based sensory gloves for sign language recognition state of the art between 2007 and 2017. Sensors, 18(7), 2208. https://doi.org/10.3390/s18072208.
- Ansari, N., Awari, S., Movva, S. H., Parthasarathy, A., & Poriwade, S. (2024, July). GesSpy: ML Driven Real Time Sign Language Detection. In 2024 5th International Conference on Image Processing and Capsule Networks (ICIPCN) (pp. 912-917). IEEE. https://doi.org/10.1109/ICIPCN63822.2024.00157.
- Camgoz, N. C., Koller, O., Hadfield, S., & Bowden, R. (2020). Sign language transformers: Joint end-to-end sign language recognition and translation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 10023-10033).
- Kawamura, M., Shirahata, Y., Yamamoto, R., & Tachibana, K. (2023, June). Lightweight and high-fidelity end-to-end text-to-speech with multi-band generation and inverse short-time fourier transform. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE. https://doi.org/10.1109/ICASSP49357.2023.10095296.
- Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., … Grundmann, M. (2019). MediaPipe: A framework for building perception pipelines. arXiv. https://arxiv.org/abs/1906.08172.
- Mederos, B., Mejía, J., Medina-Reyes, A., Espinosa-Almeyda, Y., Díaz-Roman, J. D., Rodríguez-Mederos, I., ... & Gonzalez-Lopez, F. (2025). Sign language recognition from skeletal data using graph and recurrent neural networks. arXiv preprint arXiv:2511.05772.
- Nair, M., Tanwar, S., Badotra, S., & Kukreja, V. (2023, September). Use of neural machine translation in multimodal translation. In 2023 6th International Conference on Contemporary Computing and Informatics (IC3I) (Vol. 6, pp. 130-135). IEEE. https://doi.org/10.1109/IC3I59117.2023.10397780.
- Nguyen, T. T., Nguyen, N. C., Ngo, D. K., Phan, V. L., Pham, M. H., Nguyen, D. A., ... & Le, T. L. (2022, November). A continuous real-time hand gesture recognition method based on skeleton. In 2022 11th International Conference on Control, Automation and Information Sciences (ICCAIS) (pp. 273-278). IEEE. https://doi.org/10.1109/ICCAIS56082.2022.9990122.
- Özdemir, O., Baytaş, İ. M., & Akarun, L. (2023). Multi-cue temporal modeling for skeleton-based sign language recognition. Frontiers in neuroscience, 17, 1148191.
- Pu, M., Lim, M. K., & Chong, C. Y. (2024, October). Siformer: Feature-isolated transformer for efficient skeleton-based sign language recognition. In Proceedings of the 32nd ACM International Conference on Multimedia (pp. 9387-9396).
- Sincan, O. M., & Keles, H. Y. (2020). Autsl: A large scale multi-modal turkish sign language dataset and baseline methods. IEEE access, 8, 181340-181355.
- Tu, K., Pan, C., Zhang, J., Jin, Y., Wang, J., & Shi, G. (2015, June). Improvement of chinese sign language translation system based on multi-node micro inertial measurement unit. In 2015 IEEE International Conference on Cyber Technology in Automation, Control, and Intelligent Systems (CYBER) (pp. 1781-1786). IEEE. https://doi.org/10.1109/CYBER.2015.7288216.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

---

# EKLER

## EK-1
