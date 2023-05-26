import string
import tkinter as tk
from tkinter import filedialog
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from nltk import word_tokenize
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from nltk.corpus import stopwords
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


nlp = spacy.load('en_core_web_sm')

nltk.download('popular')

# GloVe Word Embedding dosyası yükleme
embeddings_index = {}
with open('glove.6B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs


def cumle_duzenle(sentence):
    # Noktalama işaretleri
    punctuations = string.punctuation

    # Stop words
    stop_words = set(stopwords.words('english'))

    # Küçük harflere dönüştürme
    sentence = sentence.lower()

    # Noktalama işaretlerini kaldırma
    sentence = ''.join(char for char in sentence if char not in punctuations)

    # Stop wordsleri kaldırma
    words = word_tokenize(sentence)
    words = [word for word in words if word not in stop_words]
    sentence1 = " ".join(words)
    return sentence1


# Cümleleri vektör temsillerine dönüştürme
def vektor_hesap(sentence):
    words = sentence.lower().split()
    sentence_vector = np.zeros((300,), dtype='float32')
    for word in words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            sentence_vector += embedding_vector
    return sentence_vector


# Kosinüs benzerliğini hesaplama
def kosinus_benzerlik(sentence1, sentence2):
    vector1 = vektor_hesap(sentence1)
    vector2 = vektor_hesap(sentence2)
    similarity = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))
    return similarity[0][0]


def metin_duzenle(text):
    # Noktalama işaretleri
    punctuations = string.punctuation

    # Stop words
    stop_words = set(stopwords.words('english'))

    # Metni küçük harflere dönüştürme
    text = text.lower()

    # Noktalama işaretlerini kaldırma
    text = ''.join(char for char in text if char not in punctuations)

    # Kelimeleri ayırma
    words = nltk.word_tokenize(text)

    # Stop wordsleri kaldırma
    words = [word for word in words if word not in stop_words]

    # Kaldırılan stop wordsleri ve noktalama işaretlerini birleştirme
    clean_text = ' '.join(words)

    return clean_text


root = tk.Tk()
root.attributes('-fullscreen', True)
root.configure(background="#efd964")
root.title("Yazlab 2.3")

label1 = tk.Label(root, text="Benzerlik Tresholdu giriniz:")
label1.pack()
label1.configure(foreground="black",background="#efd964")

text_box = tk.Text(root, height=1, width=30)
text_box.pack()
text_box.configure(background="#eac647",foreground="black")


def dosya_sec():
    a = 0
    k = 0
    b = 0
    c = 0
    d = 0
    e = 0
    f = 0
    h = 0
    m = 0
    metin = ""

    entered_text = text_box.get("1.0", "end-1c")  # tresholdu alma
    cupoints = {}
    file_path = filedialog.askopenfilename()
    file = open(file_path, "r")
    satirlar = file.readlines()
    baslik = satirlar[0]
    baslik = baslik.split()
    print(baslik)
    satirlar.pop(0)
    paragraf = " ".join(satirlar).strip()
    clean = metin_duzenle(paragraf)
    dokumanlar = [clean]
    # TfidfVectorizer kullanarak TF-IDF vektörlerini oluşturma
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(dokumanlar)
    # TF-IDF skorlarını alma
    tfidf_scores = zip(tfidf_vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])

    # Skorları sıralama
    siralama = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)

    # En yüksek skorlu kelimeleri bulma
    en_yuksek_skorlu_kelimeler = [kelime for kelime, skor in siralama][:int(len(siralama) * 0.1)]
    print(en_yuksek_skorlu_kelimeler)

    cumleler = nltk.sent_tokenize(paragraf)
    G = nx.Graph()
    for cumle in cumleler:
        G.add_node(cumle)
        doc = nlp(cumle)
        # özel isim kontrolü
        for token in doc:
            if token.text in string.punctuation:
                k = k
            else:
                k += 1
            if token.pos_ == "PROPN":
                print(token)
                a += 1
        cupoints[cumle] = a / k
        a = 0
        k = 0
        # numerik veri kontrolü
        for token in doc:
            if token.text in string.punctuation:
                c = c
            else:
                c += 1
            if token.text.isnumeric():
                b += 1
        cupoints[cumle] += b / c
        b = 0
        c = 0

        # baslik benzerlik kontrolü
        for token in doc:
            if token.text in string.punctuation:
                e = e
            else:
                e += 1
                for kelime in baslik:
                    if token.text.lower() == kelime.lower():
                        d += 1
        cupoints[cumle] += d / e
        d = 0
        e = 0

        # TF-IDF kontrolü
        for token in doc:
            if token.text in string.punctuation:
                f = f
            else:
                f += 1
                for kelime in en_yuksek_skorlu_kelimeler:
                    if token.text.lower() == kelime.lower():
                        h += 1
        cupoints[cumle] += h / f
        f = 0
        h = 0

    key_list = list(cupoints.keys())
    val_list = list(cupoints.values())
    # Benzerlik treshold kontrolü
    key1_list = list(cupoints.keys())
    for i in range(0, len(key1_list)):
        for j in range(0, len(key1_list)):
            if key1_list[i] == key1_list[j]:
                continue
            else:
                if kosinus_benzerlik(cumle_duzenle(key1_list[i]),
                                     cumle_duzenle(key1_list[j])) > float(entered_text):
                    m += 1

        cupoints[key1_list[i]] += m / (len(cumleler) - 1)
        m = 0
    for i in range(0, len(key_list)):
        for j in range(0, len(key_list)):
            if (i == j):
                continue
            else:
                G.add_edge(key_list[i], key_list[j], color='b')
    print(cupoints.items())
    pos = nx.circular_layout(G)
    colors = nx.get_edge_attributes(G, 'color').values()
    weights = nx.get_edge_attributes(G, 'weight').values()
    for i in range(0, len(key_list)):
        for j in range(0, len(key_list)):
            if (i == j):
                continue
            else:
                print(cumle_duzenle(key_list[i]) + " ve " + cumle_duzenle(
                    key_list[j]) + " arasındaki benzerlik : " + str(
                    kosinus_benzerlik(cumle_duzenle(key_list[i]), cumle_duzenle(key_list[j]))))
                nx.draw_networkx_edge_labels(G, pos, edge_labels={
                    (key_list[i], key_list[j]): str(kosinus_benzerlik(key_list[i], key_list[j]))},
                                             font_color='red', font_size=5)
    sorted_keys = sorted(cupoints, key=cupoints.get, reverse=True)
    for i in range(0, len(sorted_keys)):
        if i < len(sorted_keys) / 2:
            metin += sorted_keys[i]
            metin += " "
    textarea.insert(tk.END, metin)

    nx.draw(G, pos,
            edge_color=colors,
            width=list(weights),
            with_labels=True,
            font_size=6,
            node_color='lightgreen')

    plt.tight_layout()
    plt.xlim(-1.3, 1.4)
    plt.ylim(-1.5, 1.5)
    plt.gcf().set_size_inches(15, 6)
    canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()
    file.close()


def rouge_hesapla():
    ozet = textarea.get("1.0", "end-1c")
    gercek_ozet = textarea2.get("1.0", "end-1c")

    rouge = Rouge()
    scores = rouge.get_scores(gercek_ozet, ozet)
    l_skor = scores[0]["rouge-l"]["f"]

    skorlar = "Rouge skoru:" + str(l_skor)
    textarea3.insert(tk.END, skorlar)
    return skorlar


label2 = tk.Label(root, text="Özet:")
label2.pack()
label2.configure(foreground="black",background="#efd964")

textarea = tk.Text(root, height=6, width=250)
textarea.pack()
textarea.configure(background="#eac647",foreground="black")


label3 = tk.Label(root, text="Gerçek özet:")
label3.pack()
label3.configure(foreground="black",background="#efd964")


textarea2 = tk.Text(root, height=6, width=250)
textarea2.pack()
textarea2.configure(background="#eac647",foreground="black")

label4 = tk.Label(root, text="Rouge Skoru:")
label4.pack()
label4.configure(foreground="black",background="#efd964")

textarea3 = tk.Text(root, height=2, width=250)
textarea3.pack()
textarea3.configure(background="#eac647",foreground="black")

calculate_button = tk.Button(root, text="Skor Hesapla", command=rouge_hesapla)
calculate_button.pack(pady=10)
calculate_button.configure(background="#ffbbcc")

button = tk.Button(root, text="Dosya Seç", command=dosya_sec)
button.pack()
button.configure(background="#ffbbcc")


root.mainloop()