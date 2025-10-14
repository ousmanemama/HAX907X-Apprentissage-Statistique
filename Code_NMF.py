#%%
'''Ce programme met en place un système de recommandation de films basé sur les notes attribuées par les utilisateurs.
Les données sont d’abord chargées et organisées sous forme d’une matrice utilisateur × film, où les valeurs manquantes sont remplacées par des zéros.
Après une normalisation des notes, le modèle applique la factorisation en matrices non négatives (NMF) pour approximer cette matrice par le produit de deux matrices :
W, qui représente les préférences latentes des utilisateurs, et H, qui décrit les caractéristiques latentes des films.
Le programme évalue ensuite plusieurs valeurs du paramètre k, qui détermine le nombre de facteurs latents à extraire, et mesure pour chacune la valeur du RMSE.
Le modèle donnant la plus faible erreur est retenu pour prédire les notes manquantes et recommander les films que chaque utilisateur pourrait apprécier.
Enfin, le programme interprète les thèmes dominants associés aux genres et visualise les profils des utilisateurs.
'''
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from collections import Counter
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from matplotlib.lines import Line2D
#%%

# 1. Chargement des données (movies, ratings)
# Exemple : MovieLens 100K

''' Dataset MovieLens "Latest Small" disponible via l'url:
  https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'''

df1 = pd.read_csv("C:/Users/User/Desktop/COURS/COURS Master 2/Apprentissage Statistique/lecture/movies.csv", sep=',', header =  0)
movie_titles = df1.set_index("movieId")["title"].to_dict()
print(df1.head(2))
df2 = pd.read_csv("C:/Users/User/Desktop/COURS/COURS Master 2/Apprentissage Statistique/lecture/ratings.csv", sep=',', header= 0)
print(df2.head(2))

# Création matrice utilisateur × film
# Chaque ligne représente un utilisateur, chaque colonne un film, et la valeur = note attribuée
df= df2.pivot(index="userId", columns="movieId", values="rating")
# global_mean = np.nanmean(df.values)
# df = df.fillna(global_mean)
assert (df.fillna(0).values >= 0).all(), "Toutes les valeurs doivent être non négatives pour NMF."
print("=== Matrice initiale ===")
print(df.head(6))

# 2. Normalisation
scaler = MinMaxScaler()
X = scaler.fit_transform(df.fillna(0).values)

# 3. Fonction de calcul du RMSE pour évaluer la reconstruction NMF
def nmf_rmse(model, X):
    """Calcule l’erreur quadratique moyenne (RMSE) entre la matrice originale et reconstruite."""
    W = model.fit_transform(X)  # Profil des utilisateurs
    H = model.components_       # Profil des films
    X_hat = np.dot(W, H)        # Matrice reconstruite
    mask = X > 0                # ne calcule l’erreur que sur les valeurs observées
    return np.sqrt(mean_squared_error(X[mask], X_hat[mask]))

# Test du modèle pour plusieurs valeurs de k avec le solveur par défaut "cd"
# Liste des valeurs de k à tester
k_values = [2, 4, 6, 8, 10]
# Dictionnaire pour stocker les résultats
results = {}
for k in k_values:
    model = NMF(n_components=k, init="nndsvda", solver="cd", beta_loss='frobenius',random_state=42, max_iter=300)
    rmse = nmf_rmse(model, X)
    results[k] = rmse
    print(f"k={k} → RMSE = {rmse:.4f}")
# Sélection du meilleur k (RMSE minimal)
best_k = min(results, key=results.get)
print(f"\n Meilleur valeur de k = {best_k} (RMSE = {results[best_k]:.4f})")

# beta_loss ='kullback-leibler' ou  beta_loss = 'itakura-saito' pour le solveur "mu"
# Vue que pour notre cas (NMF pour système de recommandation), le meilleur choix est solver='cd', beta_loss='frobenius', c'est mieux d'éviter la comparaison entre les solveurs "cd" et "mu".
# Comparaison NMF : variation de k et solveur ("cd" vs "mu")
# print("\n=== Comparaison des méthodes NMF (CD vs MU) ===")
# for k in [5, 10, 15, 20]:
#     print(f"\n--- k = {k} ---")
#     for solver in ["cd", "mu"]:
#         model = NMF(
#             n_components=k,
#             init="nndsvda",
#             solver=solver,
#             random_state=42,
#             max_iter=1000
#         )
#         rmse = nmf_rmse(model, X)
#         print(f"Solver = {solver:>2} → RMSE = {rmse:.4f}")

# Entraînement du meilleur modèle (
best_nmf = NMF(n_components= best_k, init="nndsvda", solver="cd", beta_loss='frobenius',random_state=42, max_iter=300)
W = best_nmf.fit_transform(X)
H = best_nmf.components_
# Reconstruction de la matrice de notes prédite
X_hat = np.dot(W, H)
X_pred = scaler.inverse_transform(X_hat)
pred_df = pd.DataFrame(X_pred, index=df.index, columns=df.columns)
print("\n=== Matrice reconstruite (notes prédites) ===")
print(pred_df.round(2).head(6))

# 4. Fonction de recommandation de films pour un utilisateur donné
def recommend_items(user_id, original_df, predicted_df, top_n=2):
    """Retourne les top_n films non vus avec les meilleures notes prédites."""
    user_df = original_df.loc[user_id]
    preds = predicted_df.loc[user_id]
    mask = user_df == 0
    recs = preds[mask].sort_values(ascending=False).head(top_n)
    rec_titles = [movie_titles[i] for i in recs.index if i in movie_titles]
    return rec_titles

# 5. Interprétation des composantes latentes (films & genres)
# Associe les films à leurs genres
movie_genres = df1.set_index("movieId")["genres"].to_dict()
themes = pd.DataFrame(H, columns=df.columns)
theme_summaries = []

for i, comp in themes.iterrows():
    # Top films du thème
    top_items = comp.sort_values(ascending=False).head(10)
    top_titles = [movie_titles.get(mid, f"Film {mid}") for mid in top_items.index]
    top_genres = [movie_genres.get(mid, "") for mid in top_items.index]

    # Comptage des genres dominants
    genre_list = [g for sub in top_genres for g in sub.split('|') if g]
    dominant_genres = ", ".join([g for g, _ in Counter(genre_list).most_common(3)])
    
    theme_summaries.append({
        "Thème": f"Thème {i+1}",
        "Genres dominants": dominant_genres
    })

# Création d’un DataFrame récapitulatif des thèmes
theme_df = pd.DataFrame(theme_summaries)
theme_df

# 6. Thèmes dominants par utilisateur
# Identification des 3 thèmes les plus représentatifs de chaque utilisateur
W_df = pd.DataFrame(W, index=df.index, columns=[f"Thème {i+1}" for i in range(W.shape[1])])
user_summaries = []

for user_id, row in W_df.iterrows():
    top_themes = row.sort_values(ascending=False).head(3).index.tolist()
    
    # Cherche les genres dominants associés à ces thèmes
    user_genres = theme_df.loc[
        theme_df["Thème"].isin(top_themes),
        "Genres dominants"
    ].tolist()
    
    
    user_summaries.append({
        "userId": user_id,
        "Top 3 Thèmes": ", ".join(top_themes),
    
    })

# Crée un DataFrame récapitulatif
user_df = pd.DataFrame(user_summaries)
user_df

# Détermination du thème dominant pour chaque utilisateur
dominant_theme = np.argmax(W, axis=1)   # Thème latent principal
n_themes = W.shape[1]

# Visualisation 2D des utilisateurs selon les thèmes NMF
# Attribution d'une couleur par thème
colors = cm.tab10(dominant_theme / n_themes)

# Création du graphique 2D
plt.figure(figsize=(8, 6))
plt.scatter(W[:, 0], W[:, 1],
            c=colors,
            s=40,
            edgecolor='k',
            alpha=0.8)

# la légende (une couleur = un thème)
legend_elements = [
    Line2D([0], [0], marker='o', color='w',
           label=f'Thème {i+1}',
           markerfacecolor=cm.tab10(i / n_themes),
           markersize=8)
    for i in range(n_themes)
]
plt.legend(handles=legend_elements,
           title="Thèmes dominants",
           loc='best')
# Mise en forme
plt.xlabel("Facteur 1")
plt.ylabel("Facteur 2")
plt.title("Projection 2D des utilisateurs selon les thèmes NMF")
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualisation 3D des utilisateurs
# Détermination du thème dominant pour chaque utilisateur
dominant_theme = np.argmax(W, axis=1)
n_themes = W.shape[1]
# Visualisation 3D avec couleurs par thème
# Palette de couleurs (une couleur par thème)
colors = cm.tab10(dominant_theme / n_themes)

# Création de la figure 3D
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')

# Nuage de points coloré par thème
ax.scatter(W[:, 0], W[:, 1], W[:, 2], c=colors, s=40, alpha=0.8, edgecolor='k')

# Légende 3D
legend_elements = [
    Line2D([0], [0], marker='o', color='w',
           label=f'Thème {i+1}', markerfacecolor=cm.tab10(i / n_themes), markersize=8)
    for i in range(n_themes)
]
ax.legend(handles=legend_elements, title="Thèmes dominants", loc='best')

# Titres et étiquettes d’axes
ax.set_xlabel("Facteur 1")
ax.set_ylabel("Facteur 2")
ax.set_zlabel("Facteur 3")
ax.set_title("Projection 3D des utilisateurs selon les thèmes NMF")
plt.tight_layout()
plt.show()
#%%
