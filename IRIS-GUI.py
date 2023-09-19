############################################################################
# GUI Iris Created By Matin Afzal
# https://github.com/MatinAfzal
# contact.matin@yahoo.com
############################################################################

import numpy as np
import pandas as pd
from tkinter import *
from tkinter import messagebox
from functools import partial
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

# File identity information
__author__ = 'Matin Afzal (contact.matin@yahoo.com)'
__version__ = '0.0.1'
__last_modification__ = '2023/09/17'

# Paths
iris_types_image = r"iristypes.png"

# Dataset
iris_ds = load_iris()

# Functions
def predict(modelnumber: int, dataset: np.ndarray) -> None:
    """
    modelnumber _ int
                      0, LogisticRegression
                      1, KNeighborsClassifier
                      2, SupportVectorClassification
                      3, DecisionTreeClassifier
                      4, RandomForestClassifier
    """

    modelnumber = modelnumber.get()

    # Data pre processing
    iris_df = pd.DataFrame(dataset.data)
    iris_df["class"] = dataset.target

    x = iris_df.drop(["class"], axis=1)
    y = iris_df["class"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    x_input = np.array(
        [[float(f_0_spin.get()),
          float(f_1_spin.get()),
          float(f_2_spin.get()),
          float(f_3_spin.get())]]
    )

    if np.any(x_input) == False:
        messagebox.showerror("Input error", f"Enter the x values")
        # print(iris_df.sample(5))

    # Model selection and top level result window
    elif modelnumber == "0":
        LR = LogisticRegression(max_iter=1000)
        LR.fit(x_train, y_train)
        pred = LR.predict(x_test)

        accuarcyValLabel.configure(text=str(accuracy_score(y_test, pred)))
        recallValLabel.configure(text=str(recall_score(y_test, pred, average="macro")))
        precisionValLabel.configure(text=str(precision_score(y_test, pred, average="macro")))

        messagebox.showinfo("Predict result", f"Logistic Regression predict: {check_result(LR, x_input)}")

    elif modelnumber == "1":
        KNN = KNeighborsClassifier()
        KNN.fit(x_train, y_train)
        pred = KNN.predict(x_test)

        accuarcyValLabel.configure(text=str(accuracy_score(y_test, pred)))
        recallValLabel.configure(text=str(recall_score(y_test, pred, average="macro")))
        precisionValLabel.configure(text=str(precision_score(y_test, pred, average="macro")))

        messagebox.showinfo("Predict result", f"KNeighbors Classifier predict: {check_result(KNN, x_input)}")

    elif modelnumber == "2":
        SVCR = SVC()
        SVCR.fit(x_train, y_train)
        pred = SVCR.predict(x_test)

        accuarcyValLabel.configure(text=str(accuracy_score(y_test, pred)))
        recallValLabel.configure(text=str(recall_score(y_test, pred, average="macro")))
        precisionValLabel.configure(text=str(precision_score(y_test, pred, average="macro")))

        messagebox.showinfo("Predict result", f"Support Vector Classification predict: {check_result(SVCR, x_input)}")

    elif modelnumber == "3":
        DTC = DecisionTreeClassifier()
        DTC.fit(x_train, y_train)
        pred = DTC.predict(x_test)

        accuarcyValLabel.configure(text=str(accuracy_score(y_test, pred)))
        recallValLabel.configure(text=str(recall_score(y_test, pred, average="macro")))
        precisionValLabel.configure(text=str(precision_score(y_test, pred, average="macro")))

        messagebox.showinfo("Predict result", f"Decision Tree Classifier predict: {check_result(DTC, x_input)}")

    elif modelnumber == "4":
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
    y = model.predict(x)
    if y == [0]:
        return 'setosa'
    elif y == [1]:
        return 'versicolor'
    elif y == [2]:
        return 'virginica'

# Iris GUI window init
IG = Tk()
IG.title("IRIS GUI")
IG.resizable(width=False, height=False)
IG.geometry("700x510")

# Iris image canva
img = PhotoImage(file=iris_types_image)
imc = Canvas(IG, width=600, height=224)
imc.create_image(300, 108, image=img)
imc.pack()

# Feature selection frame
featureFrame = Frame(IG)
featureFrame.pack(fill="x", side="top", ipadx=350)

## feature 0 , SepalLenght cm , label and spinbox
f_0_label = Label(featureFrame, text="SepalLength (cm)")
f_0_label.grid(padx=50, pady=0)

f_0_spin = Spinbox(featureFrame, width=10, bd=2, font="20", from_=0, to=100, increment=0.1)
f_0_spin.grid(padx=50, pady=10)

## feature 1 , SepalWidth cm , label and spinbox
f_1_label = Label(featureFrame, text="SepalWidth (cm)")
f_1_label.grid(padx=0, pady=10, row=0, column=1)

f_1_spin = Spinbox(featureFrame, width=10, bd=2, font="20", from_=0, to=100, increment=0.1)
f_1_spin.grid(padx=0, pady=10, row=1, column=1)

## feature 2 , PetalLength cm , label and spinbox
f_2_label = Label(featureFrame, text="PetalLength (cm)")
f_2_label.grid(padx=0, pady=10, row=0, column=2)

f_2_spin = Spinbox(featureFrame, width=10, bd=2, font="20", from_=0, to=100, increment=0.1)
f_2_spin.grid(padx=50, pady=10, row=1, column=2)

## feature 3 ,  PetalWidth cm , label and spinbox
f_3_label = Label(featureFrame, text="PetalWidth (cm)")
f_3_label.grid(padx=0, pady=10, row=0, column=3)

f_3_spin = Spinbox(featureFrame, width=10, bd=2, font="20", from_=0, to=100, increment=0.1)
f_3_spin.grid(padx=0, pady=10, row=1, column=3)

# Model selection frame
modelSelection = Frame(IG)
modelSelection.pack(fill="both", side="bottom", ipady=93)

# radio buttons configure
rb_vals = [("Logesic Regression", 0),
           ("KNeighbors Classifier", 1),
           ("Support Vector Classification", 2),
           ("Decision Tree Classifier", 3),
           ("Random Forest Classifier", 4)]

val = StringVar(modelSelection, "0")

for (text, value) in rb_vals:
    Radiobutton(modelSelection, text = text, variable = val,
                value = value, indicator = 0,
                background = "light blue").grid(padx=50, pady=5, sticky="w")

# Extra values
extraValues = Entry(modelSelection, width=24, bd=2, font="bold", fg="black")
extraValues.grid(padx=80, pady=0, row=0, column=2, sticky="w")
extraValuesLabel = Label(modelSelection, text="Extra values:", fg="black")
extraValuesLabel.grid(padx=0, pady=0, row=0, column=1)

# Evaluation information
accuarcyLabel = Label(modelSelection, text="Accuarcy %:")
accuarcyLabel.grid(padx=0, pady=0, row=1, column=1, sticky="w")
accuarcyValLabel = Label(modelSelection, text="0")
accuarcyValLabel.grid(padx=0, pady=0, row=1, column=2)

recallLabel = Label(modelSelection, text="Recall:")
recallLabel.grid(padx=0, pady=0, row=2, column=1, sticky="w")
recallValLabel = Label(modelSelection, text="0")
recallValLabel.grid(padx=0, pady=0, row=2, column=2)

precisionLabel = Label(modelSelection, text="Precision:")
precisionLabel.grid(padx=0, pady=0, row=3, column=1, sticky="w")
precisionValLabel = Label(modelSelection, text="0")
precisionValLabel.grid(padx=0, pady=0, row=3, column=2)

# Predict button
load_predict = partial(predict, val, iris_ds)
predictButton = Button(modelSelection, text="Predict!", font="bold", fg="red", bd=2, command=load_predict)
predictButton.grid(padx=0, pady=5, row=4, column=2)

# Program main loop
IG.mainloop()