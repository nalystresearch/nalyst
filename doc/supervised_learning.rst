Supervised Learning
===================

Nalyst provides a comprehensive suite of supervised learning algorithms for both classification and regression tasks.

All supervised learners follow the same API:
- ``train(X, y)`` - Train the model on data
- ``infer(X)`` - Make predictions
- ``infer_proba(X)`` - Get probability estimates (classifiers only)

Classification
--------------

Logistic Regression
~~~~~~~~~~~~~~~~~~~

Binary and multiclass classification using logistic regression.

.. code-block:: python

    from nalyst.learners import LogisticLearner
    
    model = LogisticLearner(
        penalty='l2',        # Regularization type
        C=1.0,               # Inverse regularization strength
        max_iter=100
    )
    model.train(X_train, y_train)
    predictions = model.infer(X_test)
    probabilities = model.infer_proba(X_test)

Decision Tree
~~~~~~~~~~~~~

Tree-based classifier that learns decision rules from data.

.. code-block:: python

    from nalyst.learners import DecisionTreeLearner
    
    model = DecisionTreeLearner(
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1
    )
    model.train(X_train, y_train)

Random Forest
~~~~~~~~~~~~~

Ensemble of decision trees for robust predictions.

.. code-block:: python

    from nalyst.learners import RandomForestLearner
    
    model = RandomForestLearner(
        n_estimators=100,
        max_depth=10,
        max_features='sqrt'
    )
    model.train(X_train, y_train)
    
    # Access feature importances
    importances = model.feature_importances_

Gradient Boosting
~~~~~~~~~~~~~~~~~

Sequential ensemble method for high accuracy.

.. code-block:: python

    from nalyst.learners import GradientBoostingLearner
    
    model = GradientBoostingLearner(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    )
    model.train(X_train, y_train)

Support Vector Machine
~~~~~~~~~~~~~~~~~~~~~~

Classification with various kernel functions.

.. code-block:: python

    from nalyst.learners import SVMLearner
    
    # Linear SVM
    model = SVMLearner(kernel='linear', C=1.0)
    
    # RBF kernel
    model = SVMLearner(kernel='rbf', C=1.0, gamma='scale')
    
    model.train(X_train, y_train)

K-Nearest Neighbors
~~~~~~~~~~~~~~~~~~~

Instance-based learning algorithm.

.. code-block:: python

    from nalyst.learners import KNeighborsLearner
    
    model = KNeighborsLearner(
        n_neighbors=5,
        weights='uniform',  # or 'distance'
        metric='euclidean'
    )
    model.train(X_train, y_train)

Naive Bayes
~~~~~~~~~~~

Probabilistic classifier based on Bayes' theorem.

.. code-block:: python

    from nalyst.learners import GaussianNBLearner, MultinomialNBLearner
    
    # For continuous features
    model = GaussianNBLearner()
    
    # For count data
    model = MultinomialNBLearner(alpha=1.0)

Regression
----------

Linear Regression
~~~~~~~~~~~~~~~~~

Standard linear regression for continuous targets.

.. code-block:: python

    from nalyst.learners import LinearLearner
    
    model = LinearLearner()
    model.train(X_train, y_train)
    
    # Access coefficients
    print(model.coef_)
    print(model.intercept_)

Ridge Regression
~~~~~~~~~~~~~~~~

Linear regression with L2 regularization.

.. code-block:: python

    from nalyst.learners import RidgeLearner
    
    model = RidgeLearner(alpha=1.0)
    model.train(X_train, y_train)

Lasso Regression
~~~~~~~~~~~~~~~~

Linear regression with L1 regularization (feature selection).

.. code-block:: python

    from nalyst.learners import LassoLearner
    
    model = LassoLearner(alpha=0.1)
    model.train(X_train, y_train)
    
    # Check selected features (non-zero coefficients)
    selected = model.coef_ != 0

Elastic Net
~~~~~~~~~~~

Combined L1 and L2 regularization.

.. code-block:: python

    from nalyst.learners import ElasticNetLearner
    
    model = ElasticNetLearner(
        alpha=0.1,
        l1_ratio=0.5  # Balance between L1 and L2
    )
    model.train(X_train, y_train)

Tree-based Regressors
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nalyst.learners import (
        DecisionTreeLearner,
        RandomForestLearner,
        GradientBoostingLearner
    )
    
    # Set task='regression' for regression
    model = RandomForestLearner(n_estimators=100, task='regression')
    model.train(X_train, y_train)

Model Selection
---------------

Cross-Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

    from nalyst.evaluation import cross_val_score
    
    scores = cross_val_score(
        model, X, y,
        cv=5,
        scoring='accuracy'  # or 'roc_auc', 'f1', 'r2', 'mse'
    )
    print(f"CV Score: {scores.mean():.4f} ± {scores.std():.4f}")

Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nalyst.automl import HyperparameterTuner
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 0.5]
    }
    
    tuner = HyperparameterTuner(
        model_class=GradientBoostingLearner,
        param_grid=param_grid,
        cv=5,
        metric='accuracy'
    )
    tuner.train(X_train, y_train)
    
    best_model = tuner.best_model_
    print(tuner.best_params_)

Evaluation Metrics
------------------

Classification Metrics
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nalyst.evaluation import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        classification_report
    )
    
    # Binary classification
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    
    # Full report
    report = classification_report(y_true, y_pred)

Regression Metrics
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nalyst.evaluation import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
