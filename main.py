import pandas as pd  # Pour manipuler les données

# Lire le fichier CSV
df = pd.read_csv("Note_etudiants.csv")

# Afficher les 5 premières lignes
print("Aperçu du tableau :")
print(df.head())

# Dimensions du tableau
print("\nNombre de lignes et colonnes :")
print(df.shape)

# Statistiques sur les données (moyenne, max, etc.)
print("\nStatistiques descriptives :")
print(df.describe())

from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

X = df[['heures_revision', 'absences', 'moyenne_avant']]
y = df['note_finale']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Précision du modèle : {score:.2f}")

prediction = model.predict([[8, 2, 10]])
print(f"Note prédite : {prediction[0]:.2f}")

import matplotlib.pyplot as plt

# Faire les prédictions sur les X_test
y_pred = model.predict(X_test)

# Tracer les vraies valeurs vs les prédictions
plt.figure(figsize=(8, 5))
plt.plot(y_test.values, label='Vraies notes', marker='o')
plt.plot(y_pred, label='Prédictions IA', marker='x')
plt.title("Comparaison : Notes réelles vs Prédictions IA")
plt.xlabel("Étudiants (dans le test)")
plt.ylabel("Note")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n Simulation manuelle :")
h = float(input("Heures de révision ? "))
a = int(input("Nombre d'absences ? "))
m = float(input("Moyenne avant ? "))

note_predite = model.predict([[h, a, m]])
print(f" Note estimée par l’IA : {note_predite[0]:.2f} / 20")
