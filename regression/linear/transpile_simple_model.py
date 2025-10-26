import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import subprocess

# Train a simple regression model and save it as regression.joblib
print("Training and dumping model...")
df = pd.read_csv("houses.csv")
X = df[["size", "nbrooms", "garden"]]
y = df["price"]
X_train, X_prod, y_train, y_prod = train_test_split(X, y, train_size=0.4)
model = LinearRegression()
model.fit(X.values, y)
joblib.dump(model, "regression.joblib")

# Load joblib model
print("Loading model from the dump...")
model = joblib.load("regression.joblib")

# Extract coefds and intercept
coef = model.coef_
intercept = model.intercept_

# Generate C code with the coefs to run a prediction on a feature vector
print("Generating C code")
c_code = f"""#include <stdio.h>

float predict(float features[{len(coef)}]) {{
    float result = {intercept:.6f};
"""

# Linear regression prediction formula
for i, c in enumerate(coef):
    c_code += f"    result += {c:.6f}f * features[{i}];\n"

c_code += """    return result;
}

int main() {
    // Example feature vector
    float features[] = {1.0, 2.0, 3.0};
    float y_pred = predict(features);
    printf("Prediction: %f\\n", y_pred);
    return 0;
}
"""

# Write to file.c
with open("fichier.c", "w") as f:
    f.write(c_code)

print("C code has been written to fichier.c")
print("Compiling C file...")

# Compile file.c using gcc
subprocess.run(["gcc", "fichier.c", "-o", "program"], check=True)

# Run the compiled program
print("Running C file...")
result = subprocess.run(["./program"], capture_output=True, text=True)

result_model = model.predict([[1.0, 2.0, 3.0]])
print("\nProgram output:")
print("\t", result.stdout)

print("\nModel prediction:")
print("\t", result_model[0])
