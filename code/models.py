import helper_funcs 
from libraries import *
from params import *

def prep_data(df, sample_type=['aed', 'naed', 'ncaed'], **kwargs):
    X = df.drop(columns=modelling_dict[sample_type]['leakage'])
    X = X.select_dtypes(include=['float64', 'int64'])
    y = df[modelling_dict[sample_type]['target']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=2, **kwargs)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, *args):
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {np.round(accuracy_score(y_test, y_pred), 3)}. Sample: {args} Model: {model.__class__.__name__}.\n")
    print(classification_report(y_test, y_pred))
    return y_pred
     

def print_top_10_features(X_train, model):
        feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance_%': model.feature_importances_ * 100}) 
        feature_importance = feature_importance.sort_values('importance_%', ascending=False)
        print(feature_importance[:10])
        print('\n')

def collect_statistics(result_collector, model, y_test, y_pred, sample_name):
    result_collector['model'].append(model.__class__.__name__)
    result_collector['sample'].append(sample_name)
    result_collector['accuracy'].append(accuracy_score(y_test, y_pred))
    result_collector['precision'].append(precision_score(y_test, y_pred, average='weighted'))
    result_collector['recall'].append(recall_score(y_test, y_pred, average='weighted'))
    result_collector['f1'].append(f1_score(y_test, y_pred, average='weighted'))

class SimpleGNNClassification(nn.Module):
    def __init__(self):
        super(SimpleGNNClassification, self).__init__()
        self.conv1 = GCNConv(2, 16)
        self.conv2 = GCNConv(16, 3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)