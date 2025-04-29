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
