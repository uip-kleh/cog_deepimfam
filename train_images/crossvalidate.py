import os
import sys
sys.path.append(os.pardir)
import json
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import keras
from math import ceil
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.losses import CategoricalFocalCrossentropy
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import tqdm 
from common.tools import SetConfig, Tools

class Draw:
    def __init__(self) -> None:
        plt.rcParams['font.family'] = 'Open Sans'
        plt.rcParams["font.size"] = 21
        plt.figure()

    def draw_history(self, result, label, fname):
        plt.figure()
        epochs = [i for i in range(len(result[label]))]
        plt.plot(epochs, result[label], label="train")
        plt.plot(epochs, result["val_" + label], label="val_" + label)
        plt.title(label)
        plt.legend()
        plt.tight_layout()
        self.save_figure_as_pdf(fname)
    
    def draw_cm(self, cm, fname, norm=False):
        print(fname)
        plt.figure()
        
        if not norm: sns.heatmap(cm, cmap=sns.color_palette("blend:#FFFFFF,#182E3E", as_cmap=True), annot=True, fmt="d")
        else: sns.heatmap(cm, cmap=sns.color_palette("blend:#FFFFFF,#182E3E", as_cmap=True), annot=False)
        plt.xlabel("Pred")
        plt.ylabel("GT")
        self.save_figure_as_pdf(fname)

    def save_figure_as_pdf(self, fname):
        plt.tight_layout()
        plt.savefig(fname, transparent=True)
        plt.cla()
        plt.clf()
        plt.close()

class DeepImFam(SetConfig):
    def __init__(self, fname, entry) -> None:
        super().__init__(fname)
        self.entry = entry
        self.results_path = self.join_path([self.results_path, self.entry])
        self.images_path = os.path.join(self.results_path, "images")
        self.images_info_path = os.path.join(self.images_path, "images_info.csv")
        self.results_path = self.join_path([self.results_path, self.model_method, "results"])
        self.metrics_path = os.path.join(self.results_path, "metrics.json")

    # +++ 交差検証用にデータを分割 +++
    # random_stateを設定しているので毎回同じ分割
    # +++
    def validate_index(self, df, labels):
        index = []
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        for train_idx, test_idx in kf.split(df, labels):
            index.append((train_idx, test_idx))
        return index
    
    def draw_history(self, i):
        fname = os.path.join(self.results_path, "history" + str(i) + ".csv")
        history = pd.read_csv(fname)

        draw = Draw()
        loss_fname = os.path.join(self.results_path, "loss" + str(i) + ".pdf")
        draw.draw_history(history, "loss", loss_fname)
        accuracy_fname = os.path.join(self.results_path, "accuracy" + str(i) + ".pdf")
        draw.draw_history(history, "accuracy", accuracy_fname)

    class ImageDataFrameGenerator:
        image_data_gen = ImageDataGenerator(
            preprocessing_function=lambda img: 1. - img / 255.,
            # rescale = 1 / 255.
        )

        def __init__(self, images_directory, x_col, y_col, target_size, batch_size=512,) -> None:
            self.images_directory = images_directory
            self.x_col = x_col
            self.y_col = y_col
            self.target_size = target_size
            self.batch_size = batch_size,

        def get_generator(self, df, shuffle=False):
            # CALC SAMPLE WEIGHT
            sample_weight = compute_sample_weight(class_weight="balanced", y=df[self.y_col])

            # SET GENERATOR
            generator = self.image_data_gen.flow_from_dataframe(
                dataframe=df,
                directory=self.images_directory,
                x_col=self.x_col,
                y_col="labels",
                shuffle=shuffle,
                seed=0,
                target_size=self.target_size,
                color_mode="grayscale",
                class_mode="categorical",
                sample_weight=sample_weight,
                batch_size=self.batch_size,
            )
            return generator

    def generate_model(self):
        model = Sequential([
            Conv2D(16, (2, 2), activation="relu", padding="same", input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(16, (2, 2), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Conv2D(32, (2, 2), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Conv2D(32, (2, 2), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (2, 2), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            # Conv2D(64, (3, 3), activation="relu", padding="same"),
            # MaxPooling2D((2, 2)),
            Flatten(),
            Dense(4096, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(4096, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(4096, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(4096, activation="relu", kernel_regularizer=l2(0.001)),
            Dense(self.NUM_CLASS, activation="softmax"),
            ])
        
        model.summary()

        model.compile(
            optimizer=keras.optimizers.Adadelta(learning_rate=.9),
            loss="categorical_crossentropy",
            # loss=CategoricalFocalCrossentropy(name="CategoricalFocal"),
            metrics=["accuracy"]
        )

        return model
        
    # +++ 交差検証 +++
    def crossvalidate(self):
        # self.NUM_CLASS = 30
        df = pd.read_csv(self.images_info_path).astype(str)
        # df = df.query("labels < 30").astype(str)
        
        index = self.validate_index(df, df["labels"])
        train_results = []
        train_scores = []
        test_results = []
        test_scores = []

        for i, (train_index, test_index) in enumerate(index):
            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]

            # OVERSAMPLING
            # sampler = RandomOverSampler(random_state=42)
            # train_df, _ = sampler.fit_resample(train_df, train_df[self.hierarchy_label])

            # SET ImageDataDrameGenerator
            image_data_frame_gen = self.ImageDataFrameGenerator(
                images_directory=self.images_path,
                x_col="image_path",
                y_col="labels",
                target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
                batch_size=self.BATCH_SIZE,
            )

            train_gen = image_data_frame_gen.get_generator(train_df, shuffle=True)
            test_gen = image_data_frame_gen.get_generator(test_df, shuffle=False)
            
            # CALLBACK
            # reduce_lr = ReduceLROnPlateau(
            #     monitor='val_loss',
            #     factor=0.1,
            #     patience=5,
            #     min_lr=1e-3,
            # )

            # モデル
            early_stopping = EarlyStopping(
                monitor="val_loss",
                min_delta=0.0,
                patience=10,
            )

            model = self.generate_model()
            history: dict = model.fit(
                train_gen,
                validation_data=test_gen,
                epochs=1000,
                callbacks=[early_stopping],
            )  
            fname = os.path.join(self.results_path, "model" + str(i) + ".h5")
            model.save(fname)

            fname = os.path.join(self.results_path, "history" + str(i) + ".csv")
            self.save_dict_as_dataframe(history.history, fname)
            draw = Draw()
            self.draw_history(i)
            
            train_gen = image_data_frame_gen.get_generator(train_df, shuffle=False)
            test_gen = image_data_frame_gen.get_generator(test_df, shuffle=False)

            train_proba = model.predict(train_gen)
            test_proba = model.predict(test_gen)
            train_pred = np.argmax(train_proba, axis=1)
            test_pred = np.argmax(test_proba, axis=1)

            train_accuracy = accuracy_score(train_gen.labels, train_pred)
            train_results.append(train_accuracy)
            train_f1 = f1_score(train_gen.labels, train_pred, average="macro")
            train_scores.append(train_f1)
            test_accuracy = accuracy_score(test_gen.labels, test_pred)
            test_results.append(test_accuracy)
            test_f1 = f1_score(test_gen.labels, test_pred, average="macro")
            test_scores.append(test_f1)

            # SAVE PROBA
            # train_fname = os.path.join(self.results_path, "train_proba_validation" + str(i) + ".csv")
            # test_fname = os.path.join(self.results_path, "test_proba_validation" + str(i) + ".csv")
            # if not os.path.exists(train_fname):
            #     self.save_dict_as_dataframe({"labels": train_gen.labels}, train_fname)
            #     self.save_dict_as_dataframe({"labels": test_gen.labels}, test_fname)
                
            # train_dict = self.load_csv_as_dict(train_fname)
            # test_dict = self.load_csv_as_dict(test_fname)
            # for j in range(self.NUM_CLASS):
            #     train_dict["-".join([self.entry, str(j)])] = train_proba[:, j]
            #     test_dict["-".join([self.entry, str(j)])] = test_proba[:, j]
            # self.save_dict_as_dataframe(train_dict, train_fname)
            # self.save_dict_as_dataframe(test_dict, test_fname)

            # Draw Confusion Matrix
            # fname = os.path.join(self.results_path, "cm_train_" + str(i) + ".pdf")
            # cm = confusion_matrix(train_gen.labels, train_pred)
            # draw.draw_cm(cm, fname)
            fname = os.path.join(self.results_path, "normed_cm_train_" + str(i) + ".pdf")
            normed_cm = confusion_matrix(train_gen.labels, train_pred, normalize="true")
            draw.draw_cm(normed_cm, fname, norm=True)
            
            # fname = os.path.join(self.results_path, "cm_test_" + str(i) + ".pdf")
            # cm = confusion_matrix(test_gen.labels, test_pred)
            # draw.draw_cm(cm, fname)
            fname = os.path.join(self.results_path, "normed_cm_test_" + str(i) + ".pdf")
            normed_cm = confusion_matrix(test_gen.labels, test_pred, normalize="true")
            draw.draw_cm(normed_cm, fname, norm=True)

        # SAVE RESULTS
        metrics = {
            "train_accuracy": train_results,
            "train_scores": train_scores,
            "test_accuracy": test_results,
            "test_F1": test_scores,
            "train_average": sum(train_results) / len(train_results),
            "trian_F1_average": sum(train_scores) / len(train_scores),
            "test_average": sum(test_results) / len(test_results),
            "test_F1_average": sum(test_scores) / len(test_scores),
        }

        fname = os.path.join(self.results_path, "validate_metrics.json")
        self.save_obj(metrics, fname)          

if __name__ == "__main__":
    config_path = "config.yaml"
    setconfig = SetConfig(config_path)

    tools = Tools()
    entries = tools.load_aaindex1_entry("../common/aaindex1_entry.txt")

    for entry in entries:
        print("+++ Train {} PGM FILE +++".format(entry))
        deepimfam = DeepImFam(config_path, entry)
        deepimfam.crossvalidate()