import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import gensim.downloader as api


# --- Extension 1 : N-grammes et Stop Words ---
def extension_ngrams_and_stopwords(corpus: list[str]):
    print("=== EXTENSION 1 : IMPACT DES N-GRAMMES ET STOP WORDS ===\n")
    
    configs = {
        "Unigrammes (avec stop words)": CountVectorizer(ngram_range=(1, 1)),
        "Unigrammes (sans stop words)": CountVectorizer(ngram_range=(1, 1), stop_words="english"),
        "Unigrammes + Bigrammes (avec stop words)": CountVectorizer(ngram_range=(1, 2)),
        "Unigrammes + Bigrammes (sans stop words)": CountVectorizer(ngram_range=(1, 2), stop_words="english"),
    }
    
    results = []
    for name, vec in configs.items():
        X = vec.fit_transform(corpus)
        features = vec.get_feature_names_out()
        vocab_size = len(features)
        
        # Mots/bigrammes les plus fréquents
        counts = np.sum(X.toarray(), axis=0)
        top_features = sorted(zip(features, counts), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join([f"'{f}' ({c})" for f, c in top_features])
        
        results.append({"Configuration": name, "Dimension (Vocabulaire)": vocab_size, "Top 3 Features": top_str})
    
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    print("\n" + "="*70 + "\n")


# --- Extension 2 : Comparaison GloVe 50d vs 100d ---
def extension_compare_glove_models():
    print("=== EXTENSION 2 : COMPARAISON GLOVE 50d VS 100d ===\n")
    
    print("Chargement des deux modèles GloVe (50d et 100d)...")
    m50 = api.load("glove-wiki-gigaword-50")
    m100 = api.load("glove-wiki-gigaword-100")
    
    pair = ("film", "movie")
    sim50 = m50.similarity(*pair)
    sim100 = m100.similarity(*pair)
    
    print(f"Similarité '{pair[0]}' - '{pair[1]}' :")
    print(f"  - GloVe 50d  : {sim50:.4f}")
    print(f"  - GloVe 100d : {sim100:.4f}\n")
    
    target = "movie"
    v50 = [f"{w} ({s:.3f})" for w, s in m50.most_similar(target, topn=3)]
    v100 = [f"{w} ({s:.3f})" for w, s in m100.most_similar(target, topn=3)]
    
    df_comp = pd.DataFrame({
        "Modèle": ["GloVe 50d", "GloVe 100d"],
        f"Top 3 Voisins de '{target}'": [", ".join(v50), ", ".join(v100)]
    })
    print(df_comp.to_string(index=False))
    print("\n" + "="*70 + "\n")
    return m100


# --- Extension 3 : Visualisation t-SNE ---
def extension_tsne_visualization(model):
    print("=== EXTENSION 3 : REDUCTION DE DIMENSION NON-LINÉAIRE (t-SNE) ===\n")
    
    # Dataset de mots plus étendu
    words = [
        "film", "movie", "cinema", "actor", "director", "script",
        "dog", "cat", "pet", "animal", "lion", "tiger",
        "car", "bus", "truck", "train", "vehicle", "plane",
        "computer", "software", "code", "algorithm", "data", "ai"
    ]
    
    valid_words = [w for w in words if w in model.key_to_index]
    vectors = np.array([model[w] for w in valid_words])
    
    # Utilisation de t-SNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=42,max_iter=1000)
    vectors_2d = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], color="darkviolet", edgecolors="black", s=90)
    
    for word, (x, y) in zip(valid_words, vectors_2d):
        plt.annotate(word, xy=(x, y), xytext=(5, 5), textcoords="offset points", fontsize=10, weight="bold")
        
    plt.title("Visualisation t-SNE d'un vocabulaire étendu (Embeddings GloVe)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("tsne_embeddings.png")
    print("Graphique t-SNE sauvegardé sous : tsne_embeddings.png")
    plt.show()


if __name__ == "__main__":
    corpus_test = [
        "i loved the movie it was amazing",
        "the movie was okay",
        "i hated the movie it was boring",
        "the plot of the movie was brilliant and super exciting",
        "an absolute waste of time extremely boring and slow",
        "i really enjoyed the acting and the amazing soundtrack"
    ]
    
    extension_ngrams_and_stopwords(corpus_test)
    glove_100 = extension_compare_glove_models()
    extension_tsne_visualization(glove_100)