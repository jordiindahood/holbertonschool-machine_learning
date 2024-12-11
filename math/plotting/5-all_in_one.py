#!/usr/bin/env python3
"""
    all_in_one.py
"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    a function that displays all of last 5 figures in one
    """
    # 0
    y0 = np.arange(0, 11) ** 3

    # 1
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    # 2
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    # 3
    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    # 4
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    fig, plots = plt.subplots(nrows=3, ncols=2)
    fig.suptitle("All in One")
    plots[0, 0].plot(np.arange(0, 11), y0, "r")
    plots[0, 0].set_xlim(0, 10)

    plots[0, 1].scatter(x1, y1, color="magenta")
    plots[0, 1].set_xlabel("Height (in)", fontsize="x-small")
    plots[0, 1].set_ylabel("Weight (lbs)", fontsize="x-small")
    plots[0, 1].set_title("Men's Height vs Weight", fontsize="x-small")

    plots[1, 0].plot(x2, y2)
    plots[1, 0].set_xlabel("Time (years)", fontsize="x-small")
    plots[1, 0].set_yscale("log")
    plots[1, 0].set_ylabel("Fraction Remaining", fontsize="x-small")
    plots[1, 0].set_title("Exponential Decay of C-14", fontsize="x-small")
    plots[1, 0].set_xlim(0, 28650)

    plots[1, 1].plot(x3, y31, "r", linestyle="dashed")
    plots[1, 1].plot(x3, y32, "g")
    plots[1, 1].set_xlabel("Time (years)", fontsize="x-small")
    plots[1, 1].set_ylabel("Fraction Remaining", fontsize="x-small")
    plots[1, 1].set_title(
        "Exponential Decay of Radioactive Elements", fontsize="x-small"
    )
    plots[1, 1].set_xlim(0, 20000)
    plots[1, 1].set_ylim(0, 1)
    plots[1, 1].legend(["C-14", "Ra-226"], fontsize="x-small")

    plots[2, 0].remove()
    plots[2, 1].remove()

    ax = fig.add_subplot(3, 1, 3)
    ax.hist(student_grades, bins=range(0, 101, 10), edgecolor="black")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.set_xlabel("Grades", fontsize="x-small")
    ax.set_ylabel("Number of Students", fontsize="x-small")
    ax.set_title("Project A", fontsize="x-small")
    ax.set_xticks(range(0, 101, 10))

    plt.tight_layout()

    plt.show()
