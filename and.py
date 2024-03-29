from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot
import pandas as pd
import logging
import os

def main(data, modelName, plotName, eta, epochs):
    df_AND = pd.DataFrame(AND)
    logging.info(f"The is the raw dataset: {df_AND}")
    X, y = prepare_data(df_AND)

    model_and = Perceptron(eta=eta, epochs=epochs)
    model_and.fit(X, y)

    _ = model_and.total_loss()

    model_and.save(filename="and.model", model_dir="model_and")

    save_plot(df_AND, model_and, filename="and.png")

if __name__ == "__main__":
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y" : [0,0,0,1]
    }
    ETA = 0.3
    EPOCHS = 10

    try:
        logging.info(f">>>>> Starting Training for {gate} >>>>>")
        main(data=AND, modelName="and.model", plotName="and.png", eta=ETA, epochs=EPOCHS)
        logging.info("<<<<< Done training <<<<<")
    except Exception as e:
        logging.exception(e)
        raise e


