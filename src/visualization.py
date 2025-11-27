import seaborn as sb
import matplotlib.pyplot as plt
import time
import numpy as np


def plot_correlation(df):
    plt.figure(figsize=(10, 8))
    sb.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def menu_plots(df):

    choices = {
        1: ("Marks Class Count Graph", lambda: sb.countplot(x='Class', data=df, order=['L', 'M', 'H'])),
        2: ("Semester-wise", lambda: sb.countplot(x='Semester', hue='Class', data=df)),
        3: ("Gender-wise", lambda: sb.countplot(x='gender', hue='Class', data=df)),
        4: ("Nationality-wise", lambda: sb.countplot(x='NationalITy', hue='Class', data=df)),
        5: ("Grade-wise", lambda: sb.countplot(x='GradeID', hue='Class', data=df)),
        6: ("Section-wise", lambda: sb.countplot(x='SectionID', hue='Class', data=df)),
        7: ("Topic-wise", lambda: sb.countplot(x='Topic', hue='Class', data=df)),
        8: ("Stage-wise", lambda: sb.countplot(x='StageID', hue='Class', data=df)),
        9: ("Absent Days-wise", lambda: sb.countplot(x='StudentAbsenceDays', hue='Class', data=df))
    }

    while True:
        print("\nPlot Menu:")
        for k in range(1, 10):
            print(f"{k}. {choices[k][0]}")
        print("10. Exit plotting")

        try:
            ch = int(input("Enter Choice: "))
        except Exception:
            print("Invalid input. Enter a number.")
            continue

        if ch == 10:
            print("Exiting plots menu.")
            break

        if ch in choices:
            print(f"Loading Graph: {choices[ch][0]}")
            time.sleep(0.5)
            plt.figure(figsize=(10, 6))
            try:
                choices[ch][1]()
                plt.title(choices[ch][0])
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print("Plotting failed:", e)
        else:
            print("Invalid choice.")
