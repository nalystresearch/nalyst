=====================================
Nalyst Documentation
=====================================

Welcome to Nalyst - A Comprehensive Machine Learning, Statistical Analysis & Deep Learning Library

Key Features
------------

* **Machine Learning**: Full suite of classifiers, regressors, and clustering (38+ algorithms)
* **Deep Learning**: PyTorch-style neural network framework with autograd (50+ layers)
* **Statistical Analysis**: Hypothesis testing, ANOVA, correlation, time series
* **AutoML**: Automatic model selection and hyperparameter tuning
* **Explainability**: SHAP, LIME, and feature importance analysis
* **Preprocessing**: Scalers, encoders, imputers, feature selection

Quick Example
-------------

.. code-block:: python

    from nalyst import learners, evaluation
    
    # Train a classifier
    model = learners.RandomForestLearner(n_estimators=100)
    model.train(X_train, y_train)
    
    # Evaluate
    y_pred = model.infer(X_test)
    accuracy = evaluation.accuracy_score(y_test, y_pred)

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   supervised_learning
   deep_learning
   statistics

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api_reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
