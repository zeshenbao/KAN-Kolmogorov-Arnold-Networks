import torch
import pandas as pd

class CreateInput():

    def __init__(self, function_folder, deepmimo=False):
        self.deepmimo = deepmimo
        if self.deepmimo:
            self.y_train = pd.read_csv(f'./datasets/{function_folder}/y_data_train.csv', index_col=None)
            self.y_test = pd.read_csv(f'./datasets/{function_folder}/y_data_test.csv', index_col=None)
            self.X_train = pd.read_csv(f'./datasets/{function_folder}/X_data_train.csv', index_col=None)
            self.X_test = pd.read_csv(f'./datasets/{function_folder}/X_data_test.csv', index_col=None)
        else:
            self.true = pd.read_csv(f'./datasets/{function_folder}/true_data.csv', index_col=None)
            self.train = pd.read_csv(f'./datasets/{function_folder}/train_data.csv', index_col=None)
            self.test = pd.read_csv(f'./datasets/{function_folder}/test_data.csv', index_col=None)
            self.validation = pd.read_csv(f'./datasets/{function_folder}/validation_data.csv', index_col=None)

    def process_data(self, df1, df2=None):
        if self.deepmimo:
            print(df1)
            X = df1.iloc[:, :].to_numpy()
            y = df2.iloc[:, :].to_numpy()
            print(X.shape)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)

        else:
            # separate features (X) and labels (y) if labels exist in the last column
            X = df1.iloc[:, :-1].to_numpy()  # Features (all columns except the last)
            y = df1.iloc[:, -1].to_numpy()  # Labels (last column)
            
            # convert to PyTorch tensors
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)  # Use torch.long if labels are integers for classification
            
        return X_tensor, y_tensor
    
    def get_tensors(self):
        # Process and return tensors for all datasets
        if self.deepmimo:
            train_tensors = self.process_data(self.X_train, self.y_train)
            test_tensors = self.process_data(self.y_test, self.y_test)
            validation_tensors = None
            true_tensors = None

        else:
            train_tensors = self.process_data(self.train)
            test_tensors = self.process_data(self.test)
            validation_tensors = self.process_data(self.validation)
            true_tensors = self.process_data(self.true)
        
        return {
            'train': train_tensors,
            'test': test_tensors,
            'validation': validation_tensors,
            'true': true_tensors,
        }