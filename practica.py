import pandas as pd

# 1. Carrega del dataset
starWars = pd.read_csv("star_wars_character_dataset.csv")

# 2. Analisi inicial del dataset
print("Instancies:", starWars["name"].size) # Per veure les instancies, seleccionem qualsevol de les columnes i fem print del size.
print("Variables:", starWars.columns.size) # Per veure variables, fem print del size de la llista de columnes.

# 3. Seleccio del dataset
print("En un principi, no son totes rellevants, per exemple, el color d'ulls")
print("Variables importants: Height, mass, colors (hair, skin, eye)  i especies")
print("Variables a eliminar: films , vehicles, starships, gender, sex")

starWars = starWars[["height","mass","hair_color", "skin_color", "eye_color", "species", "homeworld"]]

# 4.Separació de variables (X i Y)
variablesEntrada = starWars[["height","mass","hair_color", "skin_color", "eye_color", "species"]].values
print(variablesEntrada)

variablesObjectiu = starWars["homeworld"].values
print(variablesObjectiu)

# 5. Train / Test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(variablesEntrada, variablesObjectiu, test_size=0.3, random_state=0)

# 6. Construccio dels models
## DummyClassifier
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy="most_frequent", random_state=0)

## Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

# 7. Avaluacio dels models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dummyPred = dummy.fit(x_train, y_train).predict(x_test)
gnbPred = gnb.fit(x_train, y_train).predict(x_test)

print("Score")
print("Dummy", accuracy_score(y_test,dummyPred))
print("GaussianNB", accuracy_score(y_test,gnbPred))

print("Confusion Matrix")
print("Dummy", confusion_matrix(y_test,dummyPred))
print("GaussianNB", confusion_matrix(y_test,gnbPred))