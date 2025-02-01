import pandas as pd
from trap_analysis.analysis import *
from rich import print
import plotly.express as px

if __name__ == "__main__":
    # lets explore the dataset.

    # load in the traps csv
    traps = pd.read_csv("taawharanui/data/manage_traps.csv")
    print(traps.head())

    # Let's get a map of different traps
    # plot of doc200s
