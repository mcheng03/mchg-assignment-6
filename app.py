from flask import Flask, render_template, request, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import base64

app = Flask(__name__)

def generate_plots(N, mu, sigma2, S):

    # Generate a random dataset X of size N with values between 0 and 1 and a random dataset Y with normal additive error (mean mu, variance sigma^2).

    X = np.random.uniform(0, 1, N)
    Y = np.random.normal(mu, np.sqrt(sigma2), N)

    # Fit a linear regression model to X and Y

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0] 
    intercept = model.intercept_  

    # Generate a scatter plot of (X, Y) with the fitted regression line
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y)
    plt.plot(X, model.predict(X.reshape(-1, 1)), 'r-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Regression Line Equation: Y = {intercept:.2f} + {slope:.2f}X')
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # Run S simulations and create histograms of slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        X_sim = np.random.uniform(0, 1, N)
        Y_sim = np.random.normal(mu, np.sqrt(sigma2), N)

        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)

        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Plot histograms of slopes and intercepts

    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # Below code is already provided
    # Calculate proportions of more extreme slopes and intercepts
    # For slopes, we will count how many are greater than the initial slope; for intercepts, count how many are less.
    slope_more_extreme = sum(s > slope for s in slopes) / S  # Already provided
    intercept_more_extreme = sum(i < intercept for i in intercepts) / S  # Already provided

    return plot1_path, plot2_path, slope_more_extreme, intercept_more_extreme

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])

        # Generate plots and results
        plot1, plot2, slope_extreme, intercept_extreme = generate_plots(N, mu, sigma2, S)

        return render_template("index.html", plot1=plot1, plot2=plot2,
                               slope_extreme=slope_extreme, intercept_extreme=intercept_extreme)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)