import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv("smoking.csv")

X=df.drop(['SMK_stat_type_cd'],axis=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_mod = X.select_dtypes(include='object').columns
for i in var_mod:
   X[i] = le.fit_transform(X[i])
X

# Select independent and dependent variable

y=df['SMK_stat_type_cd']

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Instantiate the model
classifier = LogisticRegression()

# Fit the model
classifier.fit(X_train, y_train)


# Save the model and the scaler to pickle files
pickle.dump(classifier, open("model.pkl", "wb"))
pickle.dump(sc, open("scaler.pkl", "wb"))
