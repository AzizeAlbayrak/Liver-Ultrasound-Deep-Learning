# Liver-Ultrasound-Deep-Learning
Deep learning-based classification of liver ultrasound images (Benign, Malignant, Normal) using CNN and ResNet18 architectures with PyTorch.

# Karaciğer Ultrason Görüntülerinin Derin Öğrenme Mimarileri ile Sınıflandırılması: Kapsamlı Bir Performans Analizi

Bu proje, gürültülü ve teşhisi zor olan karaciğer ultrason görüntülerini kullanarak hastaların karaciğer durumlarını otomatik olarak üç farklı sınıfta (**İyi Huylu / Benign**, **Kötü Huylu / Malignant**, **Normal**) sınıflandırmayı amaçlamaktadır. 

Çalışma kapsamında, temel Evrişimli Sinir Ağlarından (CNN) başlayarak Transfer Öğrenme (Transfer Learning) tabanlı ResNet18 mimarisine kadar uzanan **5 farklı model versiyonu** geliştirilmiş ve performansları karşılaştırılmıştır.

---

## 🚀 Öne Çıkan Sonuçlar & Model Performansları

Proje başlangıcından itibaren uygulanan düzenlileştirme ve mimari geliştirmeler neticesinde test doğruluğunda **%12.3'lük bir artış** sağlanmıştır.

* **En Optimum Model (Versiyon 4):** Tüm katmanları tıbbi verilerle ince ayara (**True Fine-Tuning**) tabi tutulan ResNet18 modeli, **%79.6 genel test doğruluğu** ile projenin en başarılı çözümü olmuştur.
* **Klinik Güvenilirlik:** Tıbbi açıdan en kritik sınıf olan **Kötü Huylu (Malignant)** tümörlerin tespitinde Versiyon 4 modeli **%87 F1-Skoruna** ulaşmış; 95 kötü huylu test görüntüsünün 82'sini doğru teşhis etmiştir.
* **Düzenlileştirmenin Sınırları (Versiyon 5):** Aşırı geometrik veri artırımının (yüksek rotasyon vb.) ultrason görüntülerinin o hassas ve bulanık doku yapısını (speckle noise) bozabileceği ve performansı sınırlandırabileceği (%77.5 doğruluk) bilimsel olarak kanıtlanmıştır.

### 📊 Model Test Doğrulukları (Accuracy) Karşılaştırması
* **V1 (Temel CNN):** %67.3
* **V2 (Düzenlileştirilmiş & Veri Artırımlı CNN):** %78.8
* **V3 (Dondurulmuş ResNet18):** %69.4
* **V4 (Fine-Tuning Uygulanmış ResNet18):** **%79.6**
* **V5 (Aşırı Düzenlileştirilmiş ResNet18):** %77.5

---

## 🛠️ Metodoloji ve Geliştirme Aşamaları

### 📦 1. Veri Seti ve Ön İşleme
* **Kaynak:** Xu Yiming ve arkadaşları (2022) tarafından derlenen *"Annotated Ultrasound Liver Images"* veri seti (Kaggle/Zenodo üzerinden erişilebilir).
* **Ön İşleme:** İşlem maliyetini optimize etmek adına tüm görüntüler `PyTorch transforms` modülü kullanılarak **128x128** piksel boyutlarına yeniden ölçeklendirilmiş ve Tensör formatına dönüştürülmüştür.
* **Veri Artırma (Data Augmentation):** Overfitting'i engellemek amacıyla *Random HorizontalFlip*, *Random Rotation (15 derece)* ve hafif parlaklık/kontrast ayarlamaları (*ColorJitter*) uygulanmıştır.

### 🧠 2. Model Versiyonları
1. **Versiyon 1 (V1):** Sıfırdan tasarlanan temel CNN mimarisi. Sınıf dengesizliği nedeniyle İyi Huylu sınıfını öğrenmekte zorlanmıştır.
2. **Versiyon 2 (V2):** Dropout katmanları eklenmiş, veri artırımı yoğunlaştırılmış ve sınıf dengesizliğini telafi etmek için kayıp fonksiyonuna (Cross Entropy Loss) `Class Weights` entegre edilmiş CNN mimarisi.
3. **Versiyon 3 (V3):** ImageNet ile önceden eğitilmiş, evrişim katmanları dondurulmuş ResNet18. "Alan Farkı (Domain Gap)" nedeniyle ultrason dokularında kısıtlı başarı göstermiştir.
4. **Versiyon 4 (V4):** Tüm katmanların kilitleri açılarak çok düşük bir öğrenme katsayısı (`learning rate = 0.0001`) ile baştan sona eğitilen (**True Fine-Tuning**) ResNet18. **(Optimum Çözüm)**
5. **Versiyon 5 (V5):** Yüksek oranlı Dropout (%50), Weight Decay (L2) ve zorlayıcı geometrik veri artırmaları içeren, 30 Epoch boyunca eğitilmiş aşırı düzenlileştirilmiş ResNet18 modeli.

---

## 💻 Kullanılan Teknolojiler ve Bağımlılıklar

* **Programlama Dili:** Python
* **Derin Öğrenme Çatısı:** PyTorch (`torch`, `torchvision`)
* **Veri Analizi & Metrikler:** NumPy, Scikit-Learn
* **Görselleştirme:** Matplotlib, Seaborn

---
## 📜 Kaynakça

1. **Veri Seti (Kaggle & Zenodo):** Xu, Y., Bowen, Z., Xiaohong, L., Tao, W., Jinxiu, J., Shijie, W., Yufan, L., Hongjun, Z., Tong, L., Ye, S., Rui, J., Guangyu, W., Jie, R., & Ting, C. (2022). *Annotated Ultrasound Liver Images* [Data set]. Kaggle. https://www.kaggle.com/datasets/orvile/annotated-ultrasound-liver-images-dataset (Orijinal Zenodo Erişimi: https://doi.org/10.5281/zenodo.7272660).

2. **ResNet Mimarisi:** He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770-778.

3. **PyTorch Kütüphanesi:** Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 8024-8035.
