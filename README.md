# ML Assignment 2 - IEEE-CIS Fraud Detection

კონკურსში მოცემული გვაქვს ტრანზაქციების უზარმაზარი dataset და ასევე ზოგიერთი ტრანზაქციის ავტორის identity, ცალკე ცხრილში. ჩვენ უნდა ამ მონაცემებზე გავწვრთნათ მოდელი, რომელიც გაარკვევს შემოყვანილი feature-ების მიხედვით, რომელი ტრანზაქციაა fraudulent და რომელი - არა. კონკურსში მოდელი ფასდება ROC curve-ის ქვეშ ფართობით, რომელიც განსაზღვრავს ალბათობას რომ შემთხვევით შერჩეული fraudulent ტრანზაქციას უფრო მაღალ რიცხვს მიანიჭებს მოდელი, ვიდრე არა-fraudulent ტრანზაქციას. მოდელის მიერ მინიჭებული რიცხვები აჩვენებს, თუ რამდენად დარწმუნებულია მოდელი, რომ კონკრეტული მონაცემი დადებით კლასს ეკუთვნის.

### ჩემი მიდგომა პრობლემის გადასაჭრელად

პრობლემის გადასაჭრელად გადავწყვიტე, რომ თავდაპირველად `LEFT JOIN` გამეკეთებინა ტრანზაქციებისა და იდენტობის ცხრილებზე, შემდეგ ჩამეტარებინა `Cleaning` და `Feature Engineering` ამ მონაცემებზე და გამეტესტა სხვადასხვა მოდელები: `Decision Tree`, `Random Forest`, `AdaBoost`, `GradientBoost` და `XGBoost`. იმის გამო, რომ ასობით feature გააჩნია ამ ამოცანას, ხელით გადარჩევა feature-ების, როგორც ეს წინა დავალებაში იყო შეუძლებელია. ამიტომაც ხშირად ვიყენებ `Correlation Filter`-ს, `RFE`-ს და მოდელების feature-ების გადარჩევის შესაძლებლობებს, რადგან აუტომატურად შეირჩეს ყველაზე მნიშვნელოვანი feature-ები.


# რეპოზიტორის სტრუქტურა

- **model_experiment_DecisionTree.ipynb** - notebook, რომელიც შეიცავს DecisionTree მოდელის გასაწვრთნელად ყველა ნაბიჯს.

- **model_experiment_RandomForest.ipynb** - შეიცავს RandomForest მოდელის გასაწვრთნელად ყველა ნაბიჯს.

- **model_experiment_AdaBoost.ipynb** - AdaBoost მოდელის notebook.

- **model_experiment_GradientBoost.ipynb** - GradientBoost მოდელის notebook.

- **model_experiment_XGBoost.ipynb** - XGBoost მოდელის notebook.


# Feature Engineering

### კატეგორიული ცვლადების რიცხვითში გადაყვანა

კატეგორიების რიცხვითში ენკოდირებისათვის გადავწყვიტე, რომ გამომეყენებინა `Weight of Evidence(WOE)` კოდირება. ეს გადაწყვეტილება იმიტომ მივიღე, რომ კონკურსში მოცემულია კლასიფიკაციის ამოცანა და `WOE` საუკეთესოდ მუშაობს ამ შემთხვევაში, რადგან დაგენერირებულ მნიშვნელობებშია შენახული ინფორმაცია იმის შესახებ, თუ რა კორელაცია არსებობს მოცემულ კატეგორიულ მნიშვნელობასა და `target`-ს შორის. `python`-ის კოდში გამოვიყენე `category_encoders`-დან `WOEEncoder`.

### Cleaning

გადავწყვიტე, რომ თუ `NA` მნიშვნელობების პროპორციული რაოდენობა feature-ში `50%`-ს აჭარბებს, ეს feature დამედროფა. ეს ლოგიკა წინა დავალებისგან განსხვავებით გავიტანე NA Filler-ისგან განსხვავებულ კლასში და დავარქვი `DropHighNAFeatures`. აქედან გამომდინარე შემიძლია დამოუკიდებლად დავამატო `pipeline`-ში საჭიროების მიხედვით.

### NA მნიშვნელობების დამუშავება

გადავწყვიტე, რომ დარჩენილი `NA` მნიშვნელობების შესავსებად ამჯერად გამომეყენებინა მედიანა. ლოგიკა ჩავწერე კლასში `FillNAWithMedian`, რომელიც შემიძლია საჭიროებისამებრ ცალკე ჩავამატო `pipeline`-ში.

# Feature Selection

თავდაპირველად correlated feature-ების გასაცხრილად გადავწყვიყე გამომეყენებინა კორელაციის ფილტრი. ფილტრის `threshold`-ად ავიღე `80%`. შევქმენი კლასი `CorrelationFilter`, რომელიც შეგიძლიათ **model_experiment_DecisionTree.ipynb** ფაილში ნახოთ. ამ კლასის ჩამატება შესაძლებელია `pipeline`-ში საჭიროებისამებრ.

თავდაპირველად ასევე გამოვიყენე `RFE` feature selection-ისთვის `n_features_to_select`-ის მნიშვნელობას ვარჩევდი `GridSearch`-ის საშუალებით.

# Training

## Decision Tree

**MLflow**-ზე ექსპერიმენტის სახელია `DecisionTree_Training`.

**DecisionTree** მოდელზე მუშაობა თავიდან დავიწყე `RFE` და `CorrelationFilter`-ის დახმარებით. მიზნათ მქონდა დასახული, რომ შემედარებინა DecisionTree-ის პერფორმანსი ამ ორი preprocessing ნაბიჯით და მათ გარეშე. 

**MLflow**-ზე პირველი სამი run: `DecisionTree_Cleaning`, `DecisionTree_FeatureEngineering`, `DecisionTree_FeatureSelection` დალოგილია partial pipeline-ები მოდელის გარეშე. notebook-ში წერია მთლიანი კოდი.

იმისათვის, რომ ჩემი კოდი გამეტესტა და კოდში ბაგების გამო დიდი დრო არ დამეკარგა რეალურ მონაცემებზე გაწვრთნისას მთლიანი dataset-დან ავიღე პირველი 1000 ჩანაწერი და ამ ჩანაწერებზე გავტესტე უბრალოდ ჩემი კოდის სისწორე. შესაბამისად **MLflow**-ზე დალოგილია: `DecisionTree_Model_Test_1`-დან `DecisionTree_Model_Test_3`-ის ჩათვლით მოდელები, რომელთაც შედარებით მაღალი `auc` ქულა აქვთ, თუმცა საბოლოო მოდელის არჩევანში ისინი არ გამითვალისწინებია.

ამ მიდგომის გზით ადრევე აღმოვაჩინე, რომ **Kaggle** Submission-ებისთვის ითხოვდა არა უბრალოდ *0/1* prediction-ს, არამედ ალბათობებს თითოეული ტრანზაქციისთვის. რადგან **MLflow**-ზე დალოგილი მოდელის წამოღებისას `.predict_proba()` მეთოდის გამოყენება შეუძლებელი იყო, დავწერე ჩემი საკუთარი wrapper-ი: `ProbabilityModel(mlflow.pyfunc.PythonModel)`, რომელიც მეთოდი `.predict()`-ის გამოძახებისას რეალურად დააბრუნებდა ალბათობებს და არა უკვე prediction-ს. **MLflow**-ზე დავლოგე ამ wrapper-ში ჩასმული მოდელის საცდელი ვერსიებიც `DecisionTree_Model_PyFunc_1` და `DecisionTree_Model_PyFunc_2`.


#### DecisionTree_Prob_Model

https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/0/runs/26e34af9277844d49e150a7dece44cb4

მას შემდეგ რაც დავრწმუნდი, რომ ჩემი კოდი მუშაობდა გავუშვი training მთლიან მონაცემებზე `GridSearch`-ის დახმარებით შემდეგი პარამეტრებით:

```
param_grid = {
    "feature_selector__n_features_to_select": [5, 10, 20],
    "classifier__max_depth": [5, 10],
}
```

იმისათვის, რომ დავრწმუნებულიყავი, რომ მიღებული test_score ახლოს იქნებოდა რეალურ test_score-თან ყველგან ვიყენებდი `KFold Cross Validation`-ს და თითოეული მოდელის კანდიდატისთვის ვუშვებდი `3` fold-ზე.

`GridSearch`-მა **1:30 სთ** training-ის შემდეგ შეარჩია ქვემოთ მოცემული ჰიპერპარამეტრები:
```
classifier__max_depth: 10
feature_selector__n_features_to_select: 20
```

დადო შემდეგი საბოლოო ქულები:
```
mean_test_score: 0.81
mean_train_score: 0.82
```

ამ შედეგიდან გამომდინარე გამიჩნდა ეჭვი, რომ მოდელი იყო `underfitted` და არასაკმარისად კომპლექსური.

#### DecisionTree_Overfit

https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/0/runs/5689c6e2a491453bbe78d2163cf0b334

აღმოჩნდა, რომ max_depth-ის გაზრდით მოდელი გახდა overfitted.

```
auc_test_score: 0.8208344133102746
auc_train_score: 0.8633249269569131
```

ფაქტობრივად დაიწყო train set-ში მონაცემების დაზეპირება მოდელმა, რამაც გამოიწვია მისი ვარიაციის გაზრდა test set-ზე.


#### DecisionTree_Prob_NoPreprocessing
https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/0/runs/31e2ed91958f4dfe915af000d03a93b9

ამ პირველი ექსპერიმენტის შემდეგ გადავწყვიტე, რომ მომეხსნა `RFE` და `CorrelationFilter` და მათ გარეშე გამეშვა DecisionTree-ის training.


`GridSearch`-მა შეარჩია `classifier__max_depth: 10`. არამხოლოდ გაცილებით უფრო სწრაფად გაწვრთნა მოდელი, არამედ წინა მოდელის შედეგიც კი გაუმჯობესდა ამ მიდგომით:

```
mean_test_score: 0.8365177603386846
mean_train_score: 0.8472737775044035
```


#### DecisionTree_ImpFeats_NoPreprocessing

https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/0/runs/524405385115479cb13a1f25dfb60243

წინა მოდელიდან ამოვიღე `.feature_importances_` პარამეტრი, დავსორტე feature-ები importance-ის კლებადობის მიხედვით, ავირჩიე პირველი 25 და ხელახლა გავწვრთენი მოდელი მხოლოდ ამ feature-ებით. მივიღე ძალიან მსგავსი ქულები:

```
mean_test_score: 0.8340279896868816
mean_train_score: 0.8494336092248465
```

აქედან გამომდინარე დავასკვენი, რომ `max_features` ჰიპერპარამეტრის ხელოვნურად გაზრდა ვეღარ გააუმჯობესებდა ჩემს მოდელს. ასევე დავადგინე ისიც, რომ `max_depth`-ის ძალიან გაზრდისას მოდელის ვარიაცია იზრდებოდა, train score უმჯობესდებოდა, თუმცა test score უარესდებოდა შესაბამისად ყოველთვის გადაირჩეოდა `GridSearch`-ში, როგორც არასასურველი ჰიპერპარამეტრი.


#### DecisionTree_ImbLearn_ImpFeats

https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/0/runs/2781e3c59a174f93ac8c0225b09e8141

შემდგომ შევეცადე, რომ გამეუმჯობესებინა მოდელის პერფორმანსი imbalanced learn-ზე გადასვლითა და `SMOTE` oversampler-ის გამოყენებით.


შედეგი საგრძნობლად არ გაუმჯობესებულა test set-ზე.

```
mean_test_score: 0.840376626155647
mean_train_score: 0.872917241423932
validation_score: 0.8495009514852537
```

აქ ყურადღება მივაქციოთ, რომ `classifier__max_depth: 12` აირჩია `GridSearch`-მა იმის გამო, რომ სულ მცირედ აუმჯობესებდა test_score-ს, თუმცა უკვე ამ მნიშვნელობითაც კი შესამჩნევია აცდენა train_score-დან, რაც უკვე overfitting-ის ნიშანია.

#### DecisionTree_ImbLearn_With_Preprocessing

https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/0/runs/255ab7353a6a4a1b9852b94331314d96

საბოლოოდ გადავწყვიტე, რომ დამებრუნებინა `RFE` და `CorrelationFilter` კლასები pipeline-ში და გამეშვა imbalanced learn, `SMOTE`-ით.

test score ცოტათი გაუარესდა, თუმცა აცდენა train და test score-ს შორის შემცირდა:
```
auc_train_score: 0.8395134427310299
auc_test_score: 0.8317566819010804
```


ამ მონაცემების გაანალიზებით, მივედი დასკვნამდე, რომ საბოლოოდ საუკეთესო შედეგი `DecisionTree_ImbLearn_ImpFeats` მოდელმა დადო test score-ზე
```
mean_test_score: 0.840376626155647
```

აღსანიშნავია, რომ ეს შედეგი მიღებულია KFold Cross Validation-ის შედეგად და ასევე დამატებით გატესტილია ცალკე გადადებულ validation set-ზე, რომელზეც ასევე დაახლოებით `0.84` ქულა აიღო. შედეგად მჯერა, რომ Kaggle-ის submission-შიც დაახლოებით მსგავს შედეგს დადებს.

## Random Forest

## AdaBoost


## GradientBoost


## XGBoost


# MLflow Tracking

### შენახული Parameters ველები

- random_state - seed მნიშვნელობა random გადაწყვეტილებებისთვის
- na_drop_threshold - რა ზღვარს უნდა გადასცდეს NA-ების პროპორციული რაოდენობა, რათა feature დაიდროფოს
- kfold_n_splits - KFold Cross Validation-ის დროს გამოყენებული fold-ების რაოდენობა
- pipeline_type - *(Optional)* რა ტიპის pipeline-ს ვიყენებდი: default თუ imbalanced, თუ არაა მითითებული ანუ default-ს.
- oversampler - *(Optional)* რა ტიპის oversampler-ს ვიყენებდი, მაგ. smote, თუ არაა მითითებული ანუ არ გამომიყენებია oversampler.

ამ პარამეტრების გარდა, კიდევ იქნება ჩამოთვლილი, ის პარამეტრები, რომლებიც `GridSearch`-მა შეარჩია. ეს პარამეტრები უკვე არის კონკრეტულ მოდელზე დამოკიდებული.

მაგალითად:
- classifier__n_estimators - რამდენი estimator ხე აქვს მოდელს.
- classifier__max_depth - თითოეული ხის მაქსიმალური სიღრმე.
- classifier__learning_rate - რამდენად სწრაფად სწავლობს მოდელი დაშვებულ შეცდომებზე.

### შენახული Metrics მონაცემები

საუკეთესო მოდელის metrics ინფორმაცია ამოვიღე `GridSearch`-დან. აქედან ყველაზე მნიშვნელოვანი ველებია:

- mean_test_score - KFold Cross Validation-ზე გაშვების შედეგად საშუალო ქულა რა დააგროვა test-ზე.
- std_test_score - რა იყო საშუალო კვადრატული გადახრა დაწერილი ქულების test set-ზე.
- mean_train_score - საშუალო ქულა train set-ზე.
- std_train_score - საშუალო კვადრატული გადახრა train score-ის.
- validation_score - *(Optional)* ცალკე გადადებულ validation set-ზე გაშვებისას რა ქულა მიიღო მოდელმა, თუ არ არის მითითებული ანუ მხოლოდ KFold cross validation გამოვიყენე.

ასევე მითითებულია თითოეულ kfold split-ზე რა ქულა მიიღო test-სა და train set-ზე:
- split0_test_score
- split1_test_score
- split2_test_score
- split0_train_score
- split1_train_score
- split2_train_score
