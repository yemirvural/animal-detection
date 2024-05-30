## Datasets
Bu projede kullanılan veri seti, 10 farklı hayvan sınıfına ait etiketlenmiş görüntülerden oluşmaktadır: Buffalo, Cheetah, Geyik, Fil, Tilki, Jaguar, Aslan, Panda, Kaplan, Zebra.


Veri setlerini aşağıdaki bağlantılardan bulabilirsiniz:
- [Dataset 1](https://www.kaggle.com/datasets/biancaferreira/african-wildlife)
- [Dataset 2](https://www.kaggle.com/datasets/brsdincer/danger-of-extinction-animal-image-set)
- [Dataset 3](https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset )

## Kurulum
Ortamı kurmak ve uygulamayı çalıştırmak için aşağıdaki adımları izleyin:
1. Repoyu klonlayın.
    ```bash
    git clone https://github.com/yemirvural/animal-detection
    cd animal-detection
    ```

2. Bir Python sanal ortamı oluşturun.
    ``` bash
    python3 -m venv venv
    ```

3. Sanal ortamı etkinleştirin.
    - Linux ve macOS'ta:
    ``` bash
    source venv/bin/activate
    ```
    - Windows'ta:
    ``` bash
    venv\Scripts\activate
    ```

4. Bağımlılıkları yükleyin.
    ```bash
    pip install -r requirements.txt
    ```
5. Uygulamayı çalıştırın.
    ```python
    streamlit run './scripts/app.py'
    ```
