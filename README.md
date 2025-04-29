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


#### RandomForest_Prob_Model

https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/1/runs/8745feef1cf840f798d57d86c39e8f70

თავდაპირველად მოდელი გავწვრთენი შედარებით ბევრ estimator-ზე თუმცა ხელოვნურად შევზუღდე estimator-ების სიღრმე მაქსიმუმ 5-მდე. `GridSearch`-მა შეარჩია შემდეგი ჰიპერპარამეტრები:

```
classifier__max_depth: 5
classifier__n_estimators: 100
```

და მოდელმა მომცა შესაბამისი ქულები:

```
mean_train_score: 0.8554407806041788
mean_test_score: 0.8533557563035449
```

იქიდან გამომდინარე, რომ ეს ორი შედეგი საკმაოდ ახლოა ერთმანეთთან, ეჭვი მიჩნდება, რომ ჩემს მოდელს აქვს bias და შესაბამისად არის underfitted.


#### RandomForest_Prob_Model_LongRun

https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/1/runs/745a4d045d3a487993b010768c81fbe2


ეჭვების შესამოწმებლად `GridSearch` გავუშვი შემდეგი პარამეტრებით:

```
param_grid = {
    "classifier__max_depth": [7, 10],
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_features": [10, 20, 30]
}
```

`GridSearch` გაშვებული იყო თითქმის 3 საათი. ჰიპერპარამეტრები შეირჩა უკიდურესი: 
```
classifier__max_depth: 10
classifier__max_features: 30
classifier__n_estimators: 300
```

და შედეგები test და train set-ზე საგრძნობლად გაუმჯობესდა:
```
mean_test_score: 0.9010211818793185
mean_train_score: 0.920216173466
```

აქედან გამომდინარე, ჩემი ეჭვები გამართლდა, რომ თავდაპირველი მოდელი იყო underfitted.

## AdaBoost


#### AdaBoost_Prob_Model_1
https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/4/runs/ab773553dccd4c698f2ac1cb9f10995e


თავდაპირველად AdaBoost დავათრენინგე საკმაოდ default პარამეტრებით. გამოვიყენე მხოლოდ 1 სიღრმის stump-ები AdaBoost-ის ქვეხეებად.

```
classifier__n_estimators: 100
classifier__learning_rate: 1
classifier__stump_max_depth: 1
```

ამ პარამეტრებმა შეძლო საკმაოდ სოლიდური შედეგის დადება:
```
mean_test_score: 0.8825256415014202
mean_train_score: 0.8870550889208072
```


#### AdaBoost_Prob_Model_Underfit
https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/4/runs/86a38b2baf1c4a2386d74bc1b2eba773

შევამჩნიე, რომ წინა მოდელის გაწვრთნას საკმაოდ დიდი ხანი მოუნდა და დამაინტერესა, თუ რამდენად დაეცემოდა მოდელის პერფორმანსი თუ ასწრაფების მიზნით გავწვრთნიდი მხოლოდ 5 estimator-ზე.

ქულა შემცირდა:
```
mean_test_score: 0.7408871103120306
mean_train_score: 0.7412974897476383
```

თუმცა `mean_fit_time`-ის შემცირება უფრო დრამატული იყო `264.19`-დან `78.08-მდე`.


#### AdaBoost_Prob_Model_Deeper_Stumps
https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/4/runs/a68eda690943451893873b18b1d5c00d

შემდეგ ექსპერიმენტში გადავწყვიტე გამეზარდა stump-ების სიღრმე 2-მდე და დავკვირვებოდი, გამოიწვევდა თუ არა ეს overfitted მოდელს. იმისათვის, რომ სწრაფად გამეწვრთნა მოდელი ავიღე მხოლოდ 10 estimator-ი.

არათუ არ მივიღე overfitted მოდელი, არამედ უარესი ქულა მივიღე, ვიდრე სულ თავდაპირველი run:

```
mean_test_score: 0.8269639243350818
mean_train_score: 0.8283142297864686
```


#### AdaBoost_Prob_Model_2

https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/4/runs/1e10a425b3a94a01bfa29c90009827f1

შემდეგ მოდელში შევეცადე დამერეგულირებინა `learning_rate` GridSearch-ის შემდეგი პარამეტრებით:

```
param_grid = {
    "classifier__learning_rate": [1, 2],
    "classifier__n_estimators": [70],
    "classifier__stump_max_depth": [1],
}
```
ამ run-დან გავიგე, რომ testscore ფაქტობრივად განახევრდა მაღალ `learning_rate`-ზე და შესაბამისად GridSearch-მა შეარჩია `learning_rate=1`

ყველაზე დიდი პრობლემა ის იყო, რომ თითოეულ fit-ს მოუნდა დაახლოებით 12 წთ, ასეთ მარტივ მოდელზეც კი. ეს უკვე გაცილებით უარესს ხდის `AdaBoost`-ს `RandomForest`-ზე, `GradientBoost`-სა და `XGBoost`-ზე. ამიტომ გადავწყვიტე, რომ ამ run-ის შემდეგ აღარ გამეგრძელებინა `AdaBoost`-ის training.

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
