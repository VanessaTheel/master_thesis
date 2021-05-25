from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, jaccard_score, matthews_corrcoef, precision_score, recall_score, auc, precision_recall_curve, roc_auc_score
import wandb
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os.path




def roc_auc(labels, preds, labels_list=[0,1,2,3,4,5]):
    return roc_auc_score(y_true=labels, y_score=preds) #, average='macro', labels=labels_list, multi_class='ovo')

def precision(labels, preds, labels_list=[0,1,2,3,4,5]):
    return precision_score(y_true=labels, y_pred=preds) #, average='micro', labels=labels_list)

def recall(labels, preds, labels_list=[0,1,2,3,4,5]):
    return recall_score(y_true=labels, y_pred=preds) #, average='macro', labels=labels_list)


def train():

    wandb.init(project="bertweet_pola_try", entity='van-flo')
    run_name = wandb.run.name

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    # Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present, the Dataframe should contain at least two columns, with the first column is the text with type str, and the second column in the label with type int.


    # extended dataset to have more data in class "Business Model & Innovation"
    train_data = pd.read_pickle(r'E:\IR\data\pola\NEW_train_wir_alik_fg_ext 281120_pola.pkl')
    train_df = pd.DataFrame(train_data)
    n = len(train_df.index)


    eval_data = pd.read_pickle(r'E:\IR\data\pola\NEW_test_wir_alik_fg_ext 281120_pola.pkl')
    eval_df = pd.DataFrame(eval_data)



    train_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": False,
        "use_cached_eval_features": True,
        "output_dir": os.path.join(f'bertweet_pola/max accuracy', run_name, 'output__dir'),
        "best_model_dir": os.path.join(f'bertweet_pola/max accuracy', run_name, 'best_model'),
        "save_best_model": True,
        "evaluate_during_training": True,
        "evaluate_during_training_verbose": True,
        "max_seq_length": 128,
        #"num_train_epochs": wandb.config.epochs,
        "num_train_epochs": 20,
        #"evaluate_during_training_steps": n/(wandb.config.batch_size*wandb.config.gradient_accumulation),
        "evaluate_during_training_steps": n/10,
        #'save_steps': n/(wandb.config.batch_size*wandb.config.gradient_accumulation),
        #"logging_steps": n/(wandb.config.batch_size*wandb.config.gradient_accumulation),
        'save_steps': n/10,
        "logging_steps": n/10,
        #"logging_steps": 50,
        "wandb_project": "bertweet_pola_try",
        "wand_log": True,
        # "wandb_kwargs": {"name": roberta_base_issue},
        "save_model_every_epoch": True,
        "save_eval_checkpoints": True,
        "use_early_stopping": True,
        "early_stopping_metric": "accuracy",
        "early_stopping_patience": 3,  # was 5 before
        "n_gpu": 1,
        # "manual_seed": 4,
        "use_multiprocessing": False,
        #"train_batch_size": wandb.config.batch_size,
        "train_batch_size": 6,
        "eval_batch_size": 1, #was 2, but we want to make one prediction at a time ?/!
        #"fp16": True,
        #"fp16_opt_level": "O1",
        #"gradient_accumulation_steps": wandb.config.gradient_accumulation,
        "gradient_accumulation_steps": 10,
        #'learning_rate': wandb.config.learning_rate,
        'learning_rate': 0.00006694,
        "tensorboard_dir": None,
        "use_cuda": True,
        "normalization": True,
        "reprocess_input_data": True
        # "config": {
        #     "output_hidden_states": True
        # }
    }


    # Create a ClassificationModel
    model = ClassificationModel('bertweet','vinai/bertweet-base', args=train_args)

    # You can set class weights by using the optional weight argument

    # Train the model
    model.train_model(train_df, output_dir=os.path.join(f'bertweet_pola/max accuracy', run_name, 'training'),
                      eval_df=eval_df, accuracy=accuracy_score, ham_loss=hamming_loss, roc_auc=roc_auc,
                      precision=precision, recall=recall)  # pr_auc_macro = pr_auc_w
    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df, accuracy=accuracy_score, ham_loss=hamming_loss,
                                                                roc_auc=roc_auc, precision=precision, recall=recall,
                                                                output_dir=os.path.join(
                                                                    f'bertweet_pola/max accuracy', run_name,
                                                                    'eval'), verbose=True)

    # predictions, raw_outputs = model.predict(["FACEBOOK Friendly fraud: Documents show FACEBOOK CLASS A used games to make money off of kids "])

    # try:
    wrong_pred = pd.DataFrame(wrong_predictions, columns=['colummn'])
    os.mkdir(os.path.join(f'bertweet_pola/max accuracy', run_name, 'wrong_pred'))
    wrong_pred.to_csv(os.path.join(f'bertweet_pola/max accuracy', run_name, 'wrong_pred', 'wrong_pred.csv'),
                      index=False)
    model_out = pd.DataFrame(model_outputs)
    os.mkdir(os.path.join(f'bertweet_pola/max accuracy', run_name, 'model_outputs'))
    model_out.to_csv(os.path.join(f'bertweet_pola/max accuracy', run_name, 'model_outputs', 'model_outs.csv'),
                     index=False)



if __name__ == '__main__':
    # wandb.agent(sweep_id, function=train)
    train()