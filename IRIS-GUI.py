############################################################################
# GUI Iris Created By Matin Afzal
# https://github.com/MatinAfzal
# contact.matin@yahoo.com
############################################################################

import os
import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import messagebox
from functools import partial
from PIL import Image
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

# File identity information
__author__ = 'Matin Afzal (contact.matin@yahoo.com)'
__version__ = '0.0.1'
__last_modification__ = '2023/09/19'

# Paths
iris_types_image = os.path.join(os.path.dirname(__file__), "iristypes.png")

# Dataset
iris_ds = load_iris()

# Functions
def validate_input(values: list) -> bool:
    """Validate that input values are within realistic iris feature ranges"""
    ranges = [(4, 8), (2, 4), (1, 7), (0, 2.5)]  # sepal_len, sepal_width, petal_len, petal_width
    return all(ranges[i][0] <= val <= ranges[i][1] for i, val in enumerate(values))

def predict(modelnumber: int, dataset: np.ndarray) -> None:
    """
    modelnumber _ int
                      0, LogisticRegression
                      1, KNeighborsClassifier
                      2, SupportVectorClassification
                      3, DecisionTreeClassifier
                      4, RandomForestClassifier
    """

    modelnumber = int(modelnumber.get())

    # Data pre processing
    iris_df = pd.DataFrame(dataset.data)
    iris_df["class"] = dataset.target

    x = iris_df.drop(["class"], axis=1)
    y = iris_df["class"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    try:
        input_values = [float(f_0_spin.get()),
                       float(f_1_spin.get()),
                       float(f_2_spin.get()),
                       float(f_3_spin.get())]
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid numeric values")
        return

    x_input = np.array([input_values])

    if not validate_input(input_values):
        messagebox.showwarning("Input warning", "Values outside typical iris ranges:\nSepal Length: 4-8cm, Sepal Width: 2-4cm\nPetal Length: 1-7cm, Petal Width: 0-2.5cm")
        return

    # Model selection and top level result window
    if modelnumber == 0:
        LR = LogisticRegression(max_iter=1000)
        LR.fit(x_train, y_train)
        pred = LR.predict(x_test)

        accuarcyValLabel.configure(text=str(accuracy_score(y_test, pred)))
        recallValLabel.configure(text=str(recall_score(y_test, pred, average="macro")))
        precisionValLabel.configure(text=str(precision_score(y_test, pred, average="macro")))

        messagebox.showinfo("Predict result", f"Logistic Regression predict: {check_result(LR, x_input)}")

    elif modelnumber == 1:
        KNN = KNeighborsClassifier()
        KNN.fit(x_train, y_train)
        pred = KNN.predict(x_test)

        accuarcyValLabel.configure(text=str(accuracy_score(y_test, pred)))
        recallValLabel.configure(text=str(recall_score(y_test, pred, average="macro")))
        precisionValLabel.configure(text=str(precision_score(y_test, pred, average="macro")))

        messagebox.showinfo("Predict result", f"KNeighbors Classifier predict: {check_result(KNN, x_input)}")

    elif modelnumber == 2:
        SVCR = SVC()
        SVCR.fit(x_train, y_train)
        pred = SVCR.predict(x_test)

        accuarcyValLabel.configure(text=str(accuracy_score(y_test, pred)))
        recallValLabel.configure(text=str(recall_score(y_test, pred, average="macro")))
        precisionValLabel.configure(text=str(precision_score(y_test, pred, average="macro")))

        messagebox.showinfo("Predict result", f"Support Vector Classification predict: {check_result(SVCR, x_input)}")

    elif modelnumber == 3:
        DTC = DecisionTreeClassifier()
        DTC.fit(x_train, y_train)
        pred = DTC.predict(x_test)

        accuarcyValLabel.configure(text=str(accuracy_score(y_test, pred)))
        recallValLabel.configure(text=str(recall_score(y_test, pred, average="macro")))
        precisionValLabel.configure(text=str(precision_score(y_test, pred, average="macro")))

        messagebox.showinfo("Predict result", f"Decision Tree Classifier predict: {check_result(DTC, x_input)}")

    elif modelnumber == 4:
        RFC = RandomForestClassifier()
        RFC.fit(x_train, y_train)
        pred = RFC.predict(x_test)

        accuarcyValLabel.configure(text=str(accuracy_score(y_test, pred)))
        recallValLabel.configure(text=str(recall_score(y_test, pred, average="macro")))
        precisionValLabel.configure(text=str(precision_score(y_test, pred, average="macro")))

        messagebox.showinfo("Predict result", f"Random Forest Classifier predict: {check_result(RFC, x_input)}")

def check_result(model, x) -> str:
    """
    checks iris predict type
    """
    y = model.predict(x)[0]
    iris_classes = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    return iris_classes.get(y, 'unknown')

# Iris GUI window init
IG = ctk.CTk()
IG.title("IRIS GUI")
IG.resizable(width=False, height=False)
IG.geometry("700x510")

# Iris image canva
try:
    pil_img = Image.open(iris_types_image)
    img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(600, 224))
    imc = ctk.CTkLabel(IG, image=img, text="")
    imc.pack()
except Exception as e:
    messagebox.showerror("Image Error", f"Cannot load image: {e}")
    img = None

# Feature selection frame
featureFrame = ctk.CTkFrame(IG)
featureFrame.pack(fill="x", side="top", ipadx=350)

## feature 0 , SepalLenght cm , label and spinbox
f_0_label = ctk.CTkLabel(featureFrame, text="SepalLength (cm)")
f_0_label.grid(padx=50, pady=0)

f_0_spin = ctk.CTkEntry(featureFrame, width=100, font=("", 20))
f_0_spin.grid(padx=50, pady=10)

## feature 1 , SepalWidth cm , label and spinbox
f_1_label = ctk.CTkLabel(featureFrame, text="SepalWidth (cm)")
f_1_label.grid(padx=0, pady=10, row=0, column=1)

f_1_spin = ctk.CTkEntry(featureFrame, width=100, font=("", 20))
f_1_spin.grid(padx=0, pady=10, row=1, column=1)

## feature 2 , PetalLength cm , label and spinbox
f_2_label = ctk.CTkLabel(featureFrame, text="PetalLength (cm)")
f_2_label.grid(padx=0, pady=10, row=0, column=2)

f_2_spin = ctk.CTkEntry(featureFrame, width=100, font=("", 20))
f_2_spin.grid(padx=50, pady=10, row=1, column=2)

## feature 3 ,  PetalWidth cm , label and spinbox
f_3_label = ctk.CTkLabel(featureFrame, text="PetalWidth (cm)")
f_3_label.grid(padx=0, pady=10, row=0, column=3)

f_3_spin = ctk.CTkEntry(featureFrame, width=100, font=("", 20))
f_3_spin.grid(padx=0, pady=10, row=1, column=3)

# Model selection frame
modelSelection = ctk.CTkFrame(IG)
modelSelection.pack(fill="both", side="bottom", ipady=93)

# radio buttons configure
rb_vals = [("Logistic Regression", 0),
           ("KNeighbors Classifier", 1),
           ("Support Vector Classification", 2),
           ("Decision Tree Classifier", 3),
           ("Random Forest Classifier", 4)]

val = ctk.StringVar(modelSelection, "0")

for (text, value) in rb_vals:
    ctk.CTkRadioButton(modelSelection, text = text, variable = val,
                value = value, fg_color = "light blue").grid(padx=50, pady=5, sticky="w")

# Extra values
extraValues = ctk.CTkEntry(modelSelection, width=240, font=("", 14, "bold"), text_color="black")
extraValues.grid(padx=80, pady=0, row=0, column=2, sticky="w")
extraValuesLabel = ctk.CTkLabel(modelSelection, text="Extra values:", text_color="black")
extraValuesLabel.grid(padx=0, pady=0, row=0, column=1)

# Evaluation information
accuarcyLabel = ctk.CTkLabel(modelSelection, text="Accuracy %:")
accuarcyLabel.grid(padx=0, pady=0, row=1, column=1, sticky="w")
accuarcyValLabel = ctk.CTkLabel(modelSelection, text="0")
accuarcyValLabel.grid(padx=0, pady=0, row=1, column=2)

recallLabel = ctk.CTkLabel(modelSelection, text="Recall:")
recallLabel.grid(padx=0, pady=0, row=2, column=1, sticky="w")
recallValLabel = ctk.CTkLabel(modelSelection, text="0")
recallValLabel.grid(padx=0, pady=0, row=2, column=2)

precisionLabel = ctk.CTkLabel(modelSelection, text="Precision:")
precisionLabel.grid(padx=0, pady=0, row=3, column=1, sticky="w")
precisionValLabel = ctk.CTkLabel(modelSelection, text="0")
precisionValLabel.grid(padx=0, pady=0, row=3, column=2)

# Predict button
load_predict = partial(predict, val, iris_ds)
predictButton = ctk.CTkButton(modelSelection, text="Predict!", font=("", 14, "bold"), fg_color="red", command=load_predict)
predictButton.grid(padx=0, pady=5, row=4, column=2)

# Program main loop
IG.mainloop()