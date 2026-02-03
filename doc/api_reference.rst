API Reference
=============

This is the complete API reference for the Nalyst library.

Learners (nalyst.learners)
--------------------------

Classification
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``LogisticLearner``
     - Logistic regression classifier
   * - ``DecisionTreeLearner``
     - Decision tree classifier
   * - ``RandomForestLearner``
     - Random forest ensemble classifier
   * - ``GradientBoostingLearner``
     - Gradient boosting classifier
   * - ``SVMLearner``
     - Support vector machine classifier
   * - ``KNeighborsLearner``
     - K-nearest neighbors classifier
   * - ``GaussianNBLearner``
     - Gaussian Naive Bayes
   * - ``MultinomialNBLearner``
     - Multinomial Naive Bayes
   * - ``AdaBoostLearner``
     - AdaBoost ensemble classifier
   * - ``BaggingLearner``
     - Bagging ensemble classifier
   * - ``ExtraTreesLearner``
     - Extra Trees ensemble classifier
   * - ``MLPLearner``
     - Multi-layer perceptron classifier

Regression
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``LinearLearner``
     - Linear regression
   * - ``RidgeLearner``
     - Ridge regression (L2)
   * - ``LassoLearner``
     - Lasso regression (L1)
   * - ``ElasticNetLearner``
     - Elastic Net regression (L1+L2)
   * - ``SVRLearner``
     - Support vector regression
   * - ``DecisionTreeLearner``
     - Decision tree regressor
   * - ``RandomForestLearner``
     - Random forest regressor
   * - ``GradientBoostingLearner``
     - Gradient boosting regressor

Clustering (nalyst.clustering)
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``KMeansCluster``
     - K-means clustering
   * - ``DBSCANCluster``
     - DBSCAN density-based clustering
   * - ``HierarchicalCluster``
     - Hierarchical/agglomerative clustering
   * - ``GMMCluster``
     - Gaussian Mixture Model
   * - ``SpectralCluster``
     - Spectral clustering
   * - ``MeanShiftCluster``
     - Mean shift clustering
   * - ``AffinityPropagationCluster``
     - Affinity propagation clustering
   * - ``OPTICSCluster``
     - OPTICS clustering

Dimensionality Reduction (nalyst.reduction)
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``PCAReducer``
     - Principal Component Analysis
   * - ``TSNEReducer``
     - t-SNE for visualization
   * - ``UMAPReducer``
     - UMAP for dimensionality reduction
   * - ``LDAReducer``
     - Linear Discriminant Analysis
   * - ``TruncatedSVDReducer``
     - Truncated SVD / LSA
   * - ``NMFReducer``
     - Non-negative Matrix Factorization
   * - ``ICAReducer``
     - Independent Component Analysis

Transforms (nalyst.transform)
-----------------------------

Scaling
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``StandardScaler``
     - Standardize to zero mean, unit variance
   * - ``MinMaxScaler``
     - Scale to specified range
   * - ``RobustScaler``
     - Scale using median and IQR
   * - ``MaxAbsScaler``
     - Scale by maximum absolute value
   * - ``Normalizer``
     - Normalize samples to unit norm

Encoding
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``OneHotEncoder``
     - One-hot encode categorical features
   * - ``LabelEncoder``
     - Encode labels as integers
   * - ``OrdinalEncoder``
     - Encode ordinal categorical features
   * - ``TargetEncoder``
     - Target-based encoding

Imputation
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``SimpleImputer``
     - Impute with mean/median/mode
   * - ``KNNImputer``
     - K-nearest neighbors imputation
   * - ``IterativeImputer``
     - Multivariate imputation

Feature Engineering
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``PolynomialFeatures``
     - Generate polynomial features
   * - ``SelectKBest``
     - Select top K features
   * - ``VarianceThreshold``
     - Remove low-variance features
   * - ``RFE``
     - Recursive feature elimination

Evaluation (nalyst.evaluation)
------------------------------

Classification Metrics
~~~~~~~~~~~~~~~~~~~~~~

- ``accuracy_score(y_true, y_pred)``
- ``precision_score(y_true, y_pred)``
- ``recall_score(y_true, y_pred)``
- ``f1_score(y_true, y_pred)``
- ``roc_auc_score(y_true, y_proba)``
- ``average_precision_score(y_true, y_proba)``
- ``confusion_matrix(y_true, y_pred)``
- ``classification_report(y_true, y_pred)``
- ``log_loss(y_true, y_proba)``

Regression Metrics
~~~~~~~~~~~~~~~~~~

- ``mean_squared_error(y_true, y_pred)``
- ``mean_absolute_error(y_true, y_pred)``
- ``r2_score(y_true, y_pred)``
- ``mean_absolute_percentage_error(y_true, y_pred)``
- ``explained_variance_score(y_true, y_pred)``

Clustering Metrics
~~~~~~~~~~~~~~~~~~

- ``silhouette_score(X, labels)``
- ``davies_bouldin_score(X, labels)``
- ``calinski_harabasz_score(X, labels)``
- ``adjusted_rand_score(labels_true, labels_pred)``
- ``normalized_mutual_info_score(labels_true, labels_pred)``

Cross-Validation
~~~~~~~~~~~~~~~~

- ``cross_val_score(model, X, y, cv=5, scoring='accuracy')``
- ``train_test_split(X, y, test_size=0.2)``
- ``KFold(n_splits=5)``
- ``StratifiedKFold(n_splits=5)``
- ``LeaveOneOut()``

Neural Networks (nalyst.nn)
---------------------------

Core Classes
~~~~~~~~~~~~

- ``Tensor`` - Tensor with autograd
- ``Module`` - Base class for layers
- ``Parameter`` - Learnable parameter
- ``Sequential`` - Sequential container
- ``ModuleList`` - List of modules
- ``ModuleDict`` - Dictionary of modules

Layers
~~~~~~

Linear layers: ``Linear``, ``Bilinear``, ``Identity``

Convolutional: ``Conv1d``, ``Conv2d``, ``Conv3d``, ``ConvTranspose2d``

Recurrent: ``RNN``, ``LSTM``, ``GRU``, ``RNNCell``, ``LSTMCell``, ``GRUCell``

Attention: ``MultiHeadAttention``, ``TransformerEncoderLayer``, ``TransformerDecoderLayer``

Normalization: ``BatchNorm1d``, ``BatchNorm2d``, ``LayerNorm``, ``InstanceNorm2d``, ``GroupNorm``

Pooling: ``MaxPool2d``, ``AvgPool2d``, ``AdaptiveAvgPool2d``, ``GlobalAvgPool2d``

Activation: ``ReLU``, ``LeakyReLU``, ``GELU``, ``Sigmoid``, ``Tanh``, ``Softmax``, ``Swish``

Regularization: ``Dropout``, ``Dropout2d``

Embedding: ``Embedding``, ``PositionalEncoding``

Optimizers (nalyst.nn.optim)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``SGD`` - Stochastic Gradient Descent
- ``Adam`` - Adam optimizer
- ``AdamW`` - Adam with weight decay
- ``RMSprop`` - RMSprop optimizer
- ``Adagrad`` - Adagrad optimizer

Loss Functions
~~~~~~~~~~~~~~

- ``MSELoss`` - Mean squared error
- ``L1Loss`` - Mean absolute error
- ``CrossEntropyLoss`` - Cross entropy
- ``BCELoss`` - Binary cross entropy
- ``BCEWithLogitsLoss`` - BCE with sigmoid
- ``NLLLoss`` - Negative log likelihood
- ``KLDivLoss`` - KL divergence
- ``HuberLoss`` - Huber loss
- ``SmoothL1Loss`` - Smooth L1 loss

Statistics (nalyst.stats)
-------------------------

Hypothesis Tests
~~~~~~~~~~~~~~~~

- ``ttest_1samp``, ``ttest_ind``, ``ttest_rel``
- ``chi2_contingency``, ``chisquare``
- ``f_oneway``, ``f_twoway``
- ``mannwhitneyu``, ``wilcoxon``, ``kruskal``
- ``shapiro``, ``normaltest``, ``kstest``
- ``levene``, ``bartlett``

Correlation
~~~~~~~~~~~

- ``pearsonr``, ``spearmanr``, ``kendalltau``
- ``partial_corr``, ``corrmatrix``
- ``pointbiserialr``

Other
~~~~~

- ``multipletests`` - Multiple testing correction
- ``tukey_hsd``, ``games_howell`` - Post-hoc tests
- ``power_ttest``, ``power_anova`` - Power analysis

Time Series (nalyst.timeseries)
-------------------------------

- ``ARIMA`` - ARIMA model
- ``SARIMA`` - Seasonal ARIMA
- ``VAR`` - Vector autoregression
- ``SimpleExpSmoothing`` - Simple exponential smoothing
- ``HoltLinear`` - Holt's linear trend
- ``HoltWinters`` - Holt-Winters seasonal
- ``seasonal_decompose`` - Seasonal decomposition
- ``adf_test``, ``kpss_test`` - Stationarity tests
- ``acf``, ``pacf`` - Autocorrelation functions

AutoML (nalyst.automl)
----------------------

- ``AutoClassifier`` - Automatic classifier selection
- ``AutoRegressor`` - Automatic regressor selection
- ``HyperparameterTuner`` - Grid/random search
- ``AutoPipeline`` - Automatic pipeline construction

Imbalance (nalyst.imbalance)
----------------------------

- ``SMOTE`` - Synthetic minority oversampling
- ``ADASYN`` - Adaptive synthetic sampling
- ``RandomOverSampler`` - Random oversampling
- ``RandomUnderSampler`` - Random undersampling
- ``BorderlineSMOTE`` - Borderline SMOTE
- ``SMOTETomek`` - SMOTE + Tomek links cleaning

Explainability (nalyst.explainability)
--------------------------------------

- ``SHAPExplainer`` - SHAP values
- ``LIMEExplainer`` - LIME explanations
- ``permutation_importance`` - Permutation importance
- ``partial_dependence`` - Partial dependence
- ``CounterfactualExplainer`` - Counterfactual explanations

Workflow (nalyst.workflow)
--------------------------

- ``Pipeline`` - Sequential processing pipeline
- ``ColumnTransformer`` - Apply transforms to columns
- ``FeatureUnion`` - Combine feature extractors

Other Modules
-------------

- ``nalyst.glm`` - Generalized Linear Models
- ``nalyst.survival`` - Survival analysis
- ``nalyst.robust`` - Robust regression
- ``nalyst.quantile`` - Quantile regression
- ``nalyst.nonparametric`` - Nonparametric methods
- ``nalyst.gam`` - Generalized Additive Models
- ``nalyst.diagnostics`` - Regression diagnostics
