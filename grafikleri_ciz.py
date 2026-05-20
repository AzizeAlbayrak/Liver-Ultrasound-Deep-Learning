import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")
classes = ['Benign (İyi)', 'Malignant (Kötü)', 'Normal']

#veriler
modeller = {
    "V1": {
        "ad": "V1 (Temel CNN)",
        "loss": [0.763, 0.634, 0.587, 0.574, 0.524, 0.456, 0.434, 0.429, 0.384, 0.362],
        "acc": [0.586, 0.647, 0.683, 0.707, 0.717, 0.756, 0.760, 0.772, 0.795, 0.816],
        "cm": np.array([[21, 11, 7], [23, 65, 1], [6, 0, 13]])
    },
    "V2": {
        "ad": "V2 (CNN + Aug/Reg)",
        "loss": [2.166, 0.844, 0.751, 0.721, 0.688, 0.653, 0.614, 0.592, 0.556, 0.563],
        "acc": [0.521, 0.643, 0.658, 0.676, 0.684, 0.696, 0.732, 0.703, 0.753, 0.755],
        "cm": np.array([[101, 84, 15], [42, 392, 1], [13, 1, 86]])
    },
    "V3": {
        "ad": "V3 (ResNet Dondurulmuş)",
        "loss": [0.977, 0.766, 0.688, 0.654, 0.608, 0.623, 0.577, 0.578, 0.566, 0.566],
        "acc": [0.549, 0.670, 0.731, 0.724, 0.744, 0.772, 0.767, 0.784, 0.770, 0.760],
        "cm": np.array([[20, 8, 7], [25, 69, 1], [4, 0, 13]])
    },
    "V4": {
        "ad": "V4 (ResNet Fine-Tuning)",
        "loss": [0.952, 0.475, 0.256, 0.136, 0.119, 0.077, 0.052, 0.044, 0.030, 0.019],
        "acc": [0.573, 0.835, 0.925, 0.969, 0.966, 0.979, 0.991, 0.994, 0.996, 0.998],
        "cm": np.array([[22, 6, 7], [12, 82, 1], [4, 0, 13]])
    },
    "V5": {
        "ad": "V5 (ResNet + Strong Reg)",
        "loss": [0.9972, 0.6616, 0.5364, 0.4643, 0.4305, 0.3638, 0.3333, 0.2525, 0.1709, 0.1568,
                 0.1805, 0.1543, 0.1485, 0.1143, 0.1469, 0.0824, 0.0937, 0.0661, 0.0633, 0.0516,
                 0.0747, 0.0751, 0.0411, 0.0432, 0.0339, 0.0480, 0.0698, 0.1011, 0.0616, 0.0571],
        "acc": [0.5816, 0.7296, 0.7721, 0.8078, 0.8214, 0.8776, 0.8656, 0.8844, 0.9320, 0.9473,
                0.9439, 0.9541, 0.9490, 0.9575, 0.9439, 0.9745, 0.9694, 0.9796, 0.9847, 0.9779,
                0.9694, 0.9779, 0.9932, 0.9915, 0.9898, 0.9745, 0.9813, 0.9626, 0.9830, 0.9728],
        "cm": np.array([[29, 3, 3], [21, 74, 0], [6, 0, 11]])
    }
}

print("Sadece %100 gerçek verilerle grafikler çiziliyor...")

for v_key, v_data in modeller.items():
    epochs = np.arange(1, len(v_data["loss"]) + 1)

    # 1. Eğitim Eğrileri
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', color='tab:red', fontweight='bold')
    ax1.plot(epochs, v_data["loss"], color='tab:red', marker='o', markersize=4, label='Eğitim Loss')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue', fontweight='bold')
    ax2.plot(epochs, v_data["acc"], color='tab:blue', marker='s', markersize=4, label='Eğitim Acc')
    plt.title(f'{v_data["ad"]} - Eğitim Eğrisi', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{v_key}_1_Egitim.png', dpi=300)
    plt.close()

    #confusion matrix
    plt.figure(figsize=(7, 5))
    sns.heatmap(v_data["cm"], annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 14, "weight": "bold"})
    plt.title(f'{v_data["ad"]} - Confusion Matrix', fontweight='bold')
    plt.ylabel('Gerçek', fontweight='bold')
    plt.xlabel('Tahmin', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{v_key}_2_Confusion_Matrix.png', dpi=300)
    plt.close()

#genel karşılaştırma grafiği
test_dogrulugu = [67.3, 78.8, 69.4, 79.6, 77.5]
isimler = ['V1 (CNN)', 'V2 (Reg)', 'V3 (Frozen)', 'V4 (Fine-Tune)', 'V5 (Strong Reg)']
plt.figure(figsize=(9, 5))
ax = sns.barplot(x=isimler, y=test_dogrulugu, hue=isimler, palette='viridis', dodge=False, legend=False)
plt.title('Tüm Versiyonların Test Başarısı Karşılaştırması', fontweight='bold')
plt.ylabel('Test Accuracy (%)', fontweight='bold')
plt.ylim(60, 85)
for i in ax.containers:
    ax.bar_label(i, padding=3, fmt='%.1f%%', fontweight='bold')
plt.tight_layout()
plt.savefig('Tum_Modeller_Karsilastirma.png', dpi=300)
plt.close()

print("İşlem tamamlandı! Dürüst ve temiz 11 adet grafik klasörüne kaydedildi.")