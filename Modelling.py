# Imports
#region
import numpy as np
from Processing import final
from Loader import timer
import time
timing = timer(time.time())
#endregion



# Model Creation and fine tuning
#region
def random_forest(X_train, y_train, optimal=False):
    """
    Optimal Returns Untrained {'bootstrap': True, 'max_depth': 25, 'max_features': 3, 'n_estimators': 120}
    """
    from sklearn.ensemble import RandomForestRegressor
    if optimal:
        return RandomForestRegressor(n_estimators=120, max_depth=25, bootstrap=True, max_features=3)
    from sklearn.model_selection import GridSearchCV
 
    param_grid = [{'n_estimators':[60, 70, 80, 100, 120], 'max_depth':[15, 20, 25, None], 'bootstrap':[True, False], 'max_features':[None, 2, 3]}]
    forest = RandomForestRegressor()
    grid_search = GridSearchCV(forest, param_grid, cv=3, scoring="neg_mean_squared_error")
    grid_search.fit(X_train, y_train)
    timing.timer("Forest Grid Search Complete")
    final = grid_search.best_params_
    print(final)
    return grid_search.best_estimator_
    

def svm(X_train, y_train, optimal=False):
    """
    Optimal Returns Untrained {defaults}
    """
    from sklearn.svm import SVR
    if optimal:
        return SVR()
    from sklearn.model_selection import RandomizedSearchCV

    svr = SVR()

    param_grid = {'kernel':['rbf', 'sigmoid', 'poly', 'linear'], 'C':[0.8, 1.0, 1.2]}
    n_iter = 2
    rsv = RandomizedSearchCV(svr, param_grid, n_iter=n_iter, scoring="neg_mean_squared_error")
    rsv.fit(X_train, y_train)
    timing.timer("SVR Random Search Complete")
    final = rsv.best_params_
    print(final)
    return rsv.best_estimator_

def regression(X_train, y_train, optimal=False):
    """
    Optimal Returns Untrained {'alphas': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'l1_ratio': 0.0}
    """
    from sklearn.linear_model import ElasticNetCV
    if optimal:
        return ElasticNetCV(alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], l1_ratio=0.0)
    from sklearn.model_selection import GridSearchCV

    elastic_net = ElasticNetCV()
    param_grid = {'alphas':[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], 'l1_ratio':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
    grid_search = GridSearchCV(elastic_net, param_grid, scoring="neg_mean_squared_error", cv=3)
    grid_search.fit(X_train, y_train)
    timing.timer("Regression Grid Search Complete")
    print(grid_search.best_params_)
    return grid_search.best_estimator_ 


def knn(X_train, y_train, optimal=False):
    """
    Optimal Returns Untrained {'n_neighbors': 10, 'weights': 'distance'}
    """
    from sklearn.neighbors import KNeighborsRegressor
    if optimal:
        return KNeighborsRegressor(n_neighbors=10, weights='distance')
    
    from sklearn.model_selection import GridSearchCV
    
    model = KNeighborsRegressor()
    param_grid = {'n_neighbors':[2,4,6,8,10,12,14], 'weights':['uniform', 'distance']}
    grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=3)
    grid_search.fit(X_train, y_train)
    timing.timer("Neighbors Grid Search Complete")
    print(grid_search.best_params_)
    return grid_search.best_estimator_ 


#endregion


# Ensemble Techniques
#region
def ensemble(model_list):
    from sklearn.ensemble import VotingRegressor
    vtr = VotingRegressor(model_list)
    return vtr

#endregion

# Gradient Boosting Model
#region
class GradientBoost:
    def __init__(self, model_class, example, n_estimators=2):
        self.model_class = model_class
        self.parameters = example.get_params()
        self.n_estimators = n_estimators
        self.estimators = []
        for n in  range(n_estimators):
            model = model_class()
            model.set_params(**self.parameters)
            self.estimators.append(model)

    def fit_helper(self, X, y, i=0):
        if i >= len(self.estimators):
            return
        else:
            self.estimators[i].fit(X, y)
            preds = self.estimators[i].predict(X)
            error = y - preds
            self.fit_helper(X, error, i=i+1)

           
    def fit(self, X, y):
        self.fit_helper(X, y)

    def predict(self, X):
        prediction = self.estimators[0].predict(X)
        for estimator in self.estimators[1:]:
            prediction += estimator.predict(X)
        return prediction

    def score(self, X, y):
        from sklearn.metrics import r2_score
        preds = self.predict(X)
        return r2_score(y, preds)

#endregion
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = final(size=20000, reduce_dims=True)
    order = {0: "rfr", 1:"svr", 2:"lin_reg", 3:"knn", 4:"vtr", 5:"gb"}
    model_creators = [random_forest, svm, regression, knn]
    model_list = []
    models = []
    for i, creator in enumerate(model_creators):
        model_list.append( (order[i] , creator(X_train, y_train, optimal=True) ) )
        models.append(creator(X_train, y_train, optimal=True))
        timing.timer(f"Appended Model {i}")

    models.append(ensemble(model_list))
    from sklearn.svm import SVR
    models.append(GradientBoost(SVR, models[1], 2))

    scores = []
    for i, model in enumerate(models):
        model.fit(X_train, y_train)
        timing.timer(f"Finished Fitting model {i}")

    for model in models:
        scores.append(model.score(X_test, y_test))

    print(scores)
    