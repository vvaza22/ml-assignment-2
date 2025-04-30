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

მოდელები, რომლებიც გავტესტე აქვთ საკმაოდ კარგი შესაძლებლობა თავად შეარჩიონ უმნიშვნელოვანესი feature-ები training-ის პროცესში.

თავდაპირველად, როდესაც დავიწყე `DecisionTree`-ს გაწვრთნა გადავწყვიტე, რომ შევშველებოდი `CorrelationFilter`-ითა და `RFE`-ით. `CorrelationFilter`-ის `threshold`-ად ავიღე `80%`, ხოლო  `RFE` feature selection-ისთვის `n_features_to_select`-ის მნიშვნელობას ვარჩევდი `GridSearch`-ის საშუალებით. თუმცა როგორც გაირკვევა ქვემოთ მოყვანილი დეტალური training-ის აღწერით. `RFE` და `CorrelationFilter` ძალიან ანელებდა training-ის პროცესს და არც ძალიან კარგი შედეგი არ დადო `native` feature selection-თან შედარებით. ასე, რომ მომავალ მოდელებში უბრალოდ აღარ გამომიყენებია.

ასევე ზოგჯერ ვიყენებდი feature importances და shap მნიშვნელობებს იმისათვის, რომ გამეგო რომელი feature-ები იყო ყველაზე მნიშვნელოვანი მოდელისთვის. რამდენჯერმე მხოლოდ პირველი რამდენიმე important feature მოვნიშნე და ამ feature-ებით ხელახლა გავწვრთენი მოდელი. ეს იყო ერთ-ერთი მიდგომა, რომელიც მეხმარებოდა ორმხრივად: 

1. გაცილებით უფრო სწრაფი იყო training ნაკლებ feature-ზე და 
1. თუ დაახლოებით იგივე ქულას მომცემდა მოდელი, მაშინ ვხდებოდი, რომ `max_features` ჰიპერპარამეტრის გაზრდას აზრი არ აქვს.

# Training

Data Inspection-ის დროს გავარკვიე, რომ მონაცემები იყო არადაბალანსებული target-ის მიხედვით. ამიტომ კროსვალიდაციისთვის, გადავწყვიტე, რომ გამომეყენებინა `StratifiedKFold`, რომელიც განაწილებას ინარჩუნებს split-ებში. Training-ის სხვადასხვა ეტაპზე ვიყენებდი, როგორც ჩვეულებრივე `Pipeline`-ს ასევე `ImbalancedPipeline`-ს `SMOTE`-თან ერთად, რათა უფრო მეტ fraudulent ტრანზაქციაზე გამეწვრთნა ჩემი მოდელი. გამოყენებული მიდგომები აღწერილია თითოეული მოდელისთვის `MLflow`-ს `Parameters`-ში.

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


#### GradientBoost_Prob_Model_1
https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/3/runs/caa9c38abca64440af89be3484928644


როგორც სხვა შემთხვევებში, `GradientBoost`-ის გაწვრთნას ვიწყებ შედარებით მარტივი მოდელით და კომპლექსურობას ვუზრდი შემდეგ run-ებზე. თავდაპირველი run-ის პარამეტრებიდან `GridSearch`-მა ამოარჩია:

```
classifier__learning_rate: 0.1
classifier__max_depth: 3
classifier__n_estimators: 30
```

მოდელმა მომცა შემდეგი შედეგები:

```
mean_test_score: 0.8621642808214002
mean_train_score: 0.8641513980933139
```

ახლა საჭიროა გადამოწმდეს, რომ მოდელი არ არის underfitted და ეს შესაძლებელია კომპლექსურობის გაზრდით.

#### GradientBoost_Prob_Model_2
https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/3/runs/74ee43077cf041c695a468b33fb4438a


მეორე run-ისთვის საშუალებას ვაძლევ `GridSearch`-ს, რომ უფრო კომპლექსური მოდელი გაწვრთნას, ასევე თავისუფლებას ვაძლევ `learning_rate`-ის შერჩევაზე და შედეგად აირჩევა შემდეგი პარამეტრები:

```
classifier__learning_rate: 0.5
classifier__max_depth: 3
classifier__max_features: 20
classifier__n_estimators: 30
```

გაწვრთნილი მოდელი გვაძლევს შემდეგ შედეგებს:

```
mean_test_score: 0.8670516628046139
mean_train_score: 0.8714593770651501
```

#### GradientBoost_ImbLearn_Model_Underfit
https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/3/runs/a5d887ffee0640e5996bb3f8c2861abd

მნიშვნელოვანი დაკვირვებაა, რომ თუ `max_features` და `n_estimators` მნიშვნელობებს, უფრო დაბალ რიცხვებზე დავაყენებთ, მოდელში გავაჩენთ ბაიასს.
```
classifier__learning_rate: 0.5
classifier__max_depth: 3
classifier__max_features: 10
classifier__n_estimators: 15
```

იგი ვეღარ შეძლებს დაიჭიროს მონაცემების კომპლექსურობა და შედეგად ორივე score დაეცემა:
```
mean_train_score: 0.8403792454621307
mean_test_score: 0.8382784291488473
```


#### GradientBoost_ImbLearn_Model_Overfit

https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/3/runs/23f4b13c485d40209531275a2dbef356


იმასაც დავაკვირდი, რომ `max_depth` მნიშვნელობის გაზრდის შედეგად, მოდელი გაცილებით უფრო expressive ხდება.

```
classifier__learning_rate: 0.5
classifier__max_depth: 15
classifier__max_features: 30
classifier__n_estimators: 15
```

იგი აიღებს თითქმის უნაკლო შედეგს train set-ზე, თუმცა test set-ზე ქულის ვარდნა ძალიან შესამჩნევია. ამ შემთხვევაში მოდელს ახასიათებს მაღალი ვარიაცია და არის overfitted.

```
mean_train_score: 0.9925195272634829
mean_test_score: 0.9152955711907144
```


#### GradientBoost_ImbLearn_Model_Final
https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/3/runs/f239f83891f142cdbf1ba657ab86aca0


უამრავი პარამეტრის გადარჩევის შედეგად `GridSearch`-ის საშუალებით საბოლოოდ შევჯერდი მოდელის შემდეგ ჰიპერპარამეტრებზე:

```
classifier__learning_rate: 0.5
classifier__max_depth: 6
classifier__max_features: 30
classifier__n_estimators: 50
```

ამ ჰიპერპარამეტრებით მოდელმა დადო საკმაო სოლიდური შედეგი:

```
mean_train_score: 0.9136265609129025
mean_test_score: 0.8972540084569255
```

## XGBoost

`XGBoost`-ის პირველი ექსპერიმენტებიდანვე ჩანს, რომ საკმაოდ ექსპრესიული მოდელია, რომელსაც ადვილად შეუძლია overfit.

თავდაპირველ run-ებზე `XGBoost_Prob_Model`-დან დაწყებული და `XGBoost_Prob_ImbLearn_Model_5`-ით დამთავრებული ის შეცდომა დავუშვი, რომ `GridSearch`-ს საშუალებას ვაძლევდი ძალიან ბევრი ჰიპერპარამეტრი შეეცვალა თანადროულად. შესაბამისად მიღებული შედეგებიდან კარგი დასკვნები ვერ გამომქონდა. ამ შეცდომის გამოსასწორებლად გადავწყვიტე ხელახლა გამეწვრთნა `XGBoost` ისე, რომ ყველა პარამეტრი დამეფიქსირებინა და მხოლოდ 1 ან 2 პარამეტრის ცვლილების საშუალება მიმეცა `GridSearch`-ისთვის.

#### XGBoost_Retrain_Estimators
https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/2/runs/93735797cd0b4dd5a7fb165a4b05b6f6

წინა run-ების შედარებით უშედეგობიდან გამომდინარე გადავწყვიტე, რომ თავედან დამეწყო training. ახლა ავირჩიე სხვანაირი მიდგომა. თავიდან Regularization და Lambda კონსტანტები გავუტოლე 0-ს, რადგან მათ ხელი არ შეეშალა გადაწყვეტილების მიღებაში და `GridSearch` გავუშვი შემდეგ პარამეტრებზე:

```
"classifier__max_depth": [6],
"classifier__n_estimators": [10, 50, 100],
"classifier__learning_rate": [0.3]
```

ამ run-დან გამოვიტანე დასკვნა, რომ ყველაზე მაღალი test score ჰქონდა უფრო მაღალი `n_estimators` მქონე მოდელს. მნიშვნელობა `100`-ზე მოდელმა დადო საუკეთესო შედეგი:

```
validation_score: 0.9302013563127417
mean_test_score: 0.927280733448686
mean_train_score: 0.9478872045733354
```

თუმცა ისიც გასათვალისწინებელია, რომ trade-off იყო პერფორმანსაა და გაწვრთნისთვის საჭირო დროის შორის. რადგან მაღალი მნიშვნელობა უფრო დიდხანს გაწვრთნას ითხოვდა, გადავწყვიტე 100-ზე შევჩერებულიყავი.


#### XGBoost_Retrain_MaxDepth

https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/2/runs/27906199ed60497288c7585a7c4f4323

შემდეგ გადავწყვიტე, რომ დავკვირვებოდი რა გავლენა აქვს `max_depth` ჰიპერპარამეტრს, როდესაც დაფიქსირებული მაქვს `n_estimators`. `GridSearch` გავუშვი შემდეგი პარამეტრებით:

```
"classifier__max_depth": [3, 10]
```

output-დან აღმოჩნდა, რომ როდესაც `max_depth` არის `3`, მაშინ მოდელი underfitted ხდება და კარგავს ექსპრესიულობას, რომელიც საჭიროა მონაცემების კლასიფიკაციისთვის. დაკვირვბის შედეგად აღმოჩნდა, რომ რაც უფრო იზრდებოდა `max_depth` მნიშვნელობა, მით უფრო იზრდებოდა ორივე `test_score`-იც და `train_score`-იც, თუმცა აღსანიშნავია, რომ ასევე იზრდებოდა სხვაობა ამ ორ ქულას შორის, რაც უკვე overfitting-ის ნიშანია.


`GridSeach`-მა შეარჩია `classifier__max_depth: 10` და მომცა შედეგები:

```
mean_train_score: 0.9953115256817364
mean_test_score: 0.9547437513207216
validation_score: 0.9636374289611603
```

train_score თითქმის უნაკლოა, თუმცა test_score ჩამორჩება და შესაბამისად ეს მოდელი უკვე არის overfitted.


#### XGBoost_Retrain_Lambda

https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/2/runs/5d494538f8d9445da2869837c7d3da50


შემდეგი ექსპერიმენტისთვის გადავწყვიტე, რომ შემემცირებინა overfit `lambda` ჰიპერპარამეტრის დარეგულირებით, რომელიც ხელს უშლის split-ს ხეში, თუკი gain საკმარისად დიდი არ არის. `GridSearch` გავუშვი შემდეგ პარამეტრებზე:

```
"classifier__lambda": [1, 3, 5]
```

lambda-ს მაღალი მნიშვნელობა ამცირებდა, როგორც train_score-ს ასევე test_score-ს, ასე რომ `GridSearch`-მა აირჩია `lambda=3` და მიიღო საბოლოო შედეგი:

```
mean_test_score: 0.9542078723719799
mean_train_score: 0.9915227902443476
validation_score: 0.9630909076288943
```

ჩემი აზრით, მაინც არ არის იდეალური შედეგი, ამიტომაც გადავწყვიტე `max_depth` 6-ზე დამებრუნებინა და `lambda=3` დამეტოვა.

#### XGBoost_Retrain_Lambda_Final

https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/2/runs/ee12bc86013845078be62a0ef3317580

ეს არის მიღებული შედეგი `max_depth: 6, lambda: 3` ჰიპერპარამეტრებით:

```
mean_test_score: 0.9261029631238396
mean_train_score: 0.9457706786630521
validation_score: 0.9287962155289229
```

test_score და train_score დაახლოვდა და overfitting-იც ნაკლებად შესამჩნევია.


#### XGBoost_Retrain_Regularization
https://dagshub.com/vvaza22/ml-assignment-2.mlflow/#/experiments/2/runs/00ad88b45697450ebb88d8d1867f1734

იმისათვის, რომ overfit მომეშორებინა საბოლოოდ მივმართე `Regularization`-ს:

```
"classifier__gamma": [0.1, 5, 10],
"classifier__alpha": [0.1, 5, 10]
```

`GridSearch`-მა საბოლოოდ პარამეტრებად შეარჩია:

```
classifier__alpha: 5
classifier__gamma: 0.1
```

საბოლოო შედეგია:
```
mean_train_score: 0.9441610796704234
mean_test_score: 0.9258656855276309
```

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
