import joblib
from numpy import dtype
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import subprocess
import numpy as np

# You can change this vector to test other predictions
PREDICT = [100.0, 1.0, 1.0]
predict_string = [str(x) + "f" for x in PREDICT]

# Train a simple regression model and save it as regression.joblib
print("Training and dumping model...")
df = pd.read_csv("houses.csv")
X = df[["size", "nbrooms", "garden"]]
y = df["price"] > np.median(df["price"]).astype(dtype(int))
X_train, X_prod, y_train, y_prod = train_test_split(X, y, train_size=0.4)
model = LogisticRegression()
model.fit(X.values, y)
joblib.dump(model, "regression.joblib")

# Load joblib model
print("Loading model from the dump...")
model = joblib.load("regression.joblib")

# Extract coefds and intercept
coef = model.coef_[0]
intercept = model.intercept_[0]

# Generate C code with the coefs to run a prediction on a feature vector
print("Generating C code")
c_code = """#include <stdio.h>

long double factorial(int x) {
    if (x <= 1) return 1.0;
    long double prod = 1.0;
    for (int i = 2; i <= x; i++) prod *= i;
    return prod;
}

long double power(long double x, int i) {
    long double prod = 1.0;
    while(i--) prod *= x;
    return prod;
}

long double exp_approx(long double x, int n_term) {
    if (x < -1) return 0.0;
    long double sum = 1.0;
    for (int i = 1; i <= n_term; i++) {
        sum += power(x, i) / factorial(i);
    }
    return sum;
}


// Function to compute sigmoid
float sigmoid(float x) {
    // Frame x to avoid overflow
    x = x < -10 ? -10 : (x > 10 ? 10 : x);
    return 1 / (1 + exp_approx(-x, 10));
}

"""

c_code += f"""

float predict(float features[{len(coef)}]) {{
    float result = {intercept:.6f};
"""

# Linear regression prediction formula
for i, c in enumerate(coef):
    c_code += f"    result += {c:.6f}f * features[{i}];\n"

c_code += (
    """    return sigmoid(result);

}
int main() {
    // Example feature vector
    float features[] = {"""
    + f"""{", ".join(predict_string)}"""
    + """}; 
    float y_pred = predict(features);
    printf("Prediction: %f\\n", y_pred);
    return 0;
}
"""
)

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

result_model = model.predict([PREDICT])
print("\nProgram output:")
print("\t", result.stdout)

print("\nModel prediction:")
print("\t", 1 if result_model[0] else 0)
