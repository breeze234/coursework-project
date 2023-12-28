import pandas as pd

# 1. 데이터 불러오기
data = pd.read_csv("train.csv")
print(data)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 데이터 준비 #### RandomForestModel
X = data.drop("class", axis=1)
y = data["class"]

# RandomForestClassifier를 사용한 예시
model = RandomForestClassifier()

# 각 특성을 하나씩 평가
best_feature = None
best_score = 0

for feature in X.columns:
    # 특정 특성만 선택
    X_selected = X[feature].values.reshape(-1, 1)
    
    # 교차 검증을 통해 성능 측정
    scores = cross_val_score(model, X_selected, y, cv=5)  # 5-폴드 교차 검증 사용
    average_score = scores.mean()
    
    # 가장 좋은 성능을 갖는 특성 업데이트
    if average_score > best_score:
        best_score = average_score
        best_feature = feature

print("Best Feature:", best_feature)
print("Best Score:", best_score)
--> Best Feature: troponin / Best Score: 0.8710900473933648

---- #### KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 데이터 불러오기
data = pd.read_csv("train.csv")

# troponin 열 선택
X = data[["troponin"]]
y = data["class"]

# 데이터를 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_accuracy = 0
best_k = 0

# 다양한 K 값에 대한 반복
for k in range(1, 11):  # 예시로 1부터 10까지 K 값을 평가
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"K = {k}, Accuracy = {accuracy}")

    # 가장 높은 정확도와 그에 해당하는 K 값을 업데이트
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f"Best K: {best_k}, Best Accuracy: {best_accuracy}")
-> K = 1, Accuracy = 0.8388625592417062
K = 2, Accuracy = 0.8909952606635071
K = 3, Accuracy = 0.8578199052132701
K = 4, Accuracy = 0.8957345971563981
K = 5, Accuracy = 0.8909952606635071
K = 6, Accuracy = 0.8957345971563981
K = 7, Accuracy = 0.8625592417061612
K = 8, Accuracy = 0.8957345971563981
K = 9, Accuracy = 0.8957345971563981
K = 10, Accuracy = 0.8957345971563981
Best K: 4, Best Accuracy: 0.8957345971563981


---- ####  DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv("train.csv")

# troponin 열 선택
X = data[["troponin"]]
y = data["class"]

# 데이터를 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 의사결정나무 모델 생성 및 학습
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 의사결정나무 시각화
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=["troponin"])
plt.show()

# 모델을 사용하여 예측
y_pred = model.predict(X_test)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
-> Accuracy: 0.8957345971563981
              precision    recall  f1-score   support

           0       0.80      1.00      0.89        87
           1       1.00      0.82      0.90       124

    accuracy                           0.90       211
   macro avg       0.90      0.91      0.90       211
weighted avg       0.92      0.90      0.90       211

Confusion Matrix:
[[ 87   0]
 [ 22 102]]


---- #### KneighborsClassifier 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 데이터 불러오기
data = pd.read_csv("train.csv")

# 2개의 x 값(특성) 선택
X = data[["troponin", "glucose"]]
y = data["class"]

# 데이터를 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN 모델 생성 및 학습
k = 3  # 이웃의 수, 적절한 값을 선택하세요
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# 모델을 사용하여 예측
y_pred = model.predict(X_test)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
-> Accuracy: 0.7203791469194313
              precision    recall  f1-score   support

           0       0.66      0.66      0.66        87
           1       0.76      0.77      0.76       124

    accuracy                           0.72       211
   macro avg       0.71      0.71      0.71       211
weighted avg       0.72      0.72      0.72       211

Confusion Matrix:
[[57 30]
 [29 95]]


---- Correlation 
import pandas as pd

# 데이터 불러오기
data = pd.read_csv("train.csv")

# 특성과 클래스 간의 상관 관계 계산
correlation_matrix = data.corr()
correlation_with_class = correlation_matrix["class"].abs().sort_values(ascending=False)

# class와 가장 관련이 높은 2개의 특성 선택
selected_features = correlation_with_class.index[1:3]  # class 열 제외

# 선택된 특성 출력
print("Selected Features:", selected_features)

---> Selected Features: Index(['age', 'troponin'], dtype='object')


-----------------------------RE
import pandas as pd

data = pd.read_csv("train.csv")
print(data)
-> [1055 rows x 9 columns]

x = data.iloc[:,0:8]
y = data.iloc[:,-1]

print(x)
print(y)
-> age  gender  impluse  pressurehight  pressurelow  glucose      kcm  \
0      70       1       87            141           81    106.0    0.929   
1      65       1       76            133           75    125.0    4.570   
2      60       1       80            135           75     94.0  147.400   
3      63       1       64            122           60    188.0    2.190   
4      65       1       60            129           55    117.0    1.900   
...   ...     ...      ...            ...          ...      ...      ...   
1050   41       1       66            105           59    162.0    0.515   
1051   60       1       72            113           64    161.0    2.930   
1052   60       1       66            160           83    234.0    1.220   
1053   56       1       57            110           60    101.0    4.050   
1054   60       1       89             91           51    115.0    2.560   

      troponin  
0        1.150  
1        0.549  
2        3.850  
3        0.046  
4        0.078  
...        ...  
1050     0.003  
1051     0.014  
1052     0.203  
1053     0.017  
1054     0.099  

[1055 rows x 8 columns]
0       1
1       1
2       1
3       1
4       1
       ..
1050    0
1051    0
1052    1
1053    1
1054    1
Name: class, Length: 1055, dtype: int64
-----
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42)
# print(x_train.shape, y_train.shape)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(x_train, y_train)
model.score(x_test, y_test)
-> 0.5946969696969697

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state = 42)
dt.fit(x_train, y_train)
print("학습 데이터셋", dt.score(x_train, y_train))
print("시험 데이터셋", dt.score(x_test, y_test))
-> 학습 데이터셋 1.0
시험 데이터셋 0.9848484848484849

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state = 42)
dt.fit(x_train, y_train)
print("학습 데이터셋", dt.score(x_train, y_train))
print("시험 데이터셋", dt.score(x_test, y_test))
-> 학습 데이터셋 1.0
시험 데이터셋 0.9848484848484849

# test.csv 파일로 시험해보기
t2 = dt.predict(data2)
print(t2)
-> [1 1 0 1 1 1 0 1 1 1 1 0 1 1 0 1 1 1 1 0 1 0 0 0 1 0 1 0 0 0 1 1 0 1 1 0 0
 0 0 0 0 0 0 0 0 1 0 1 0 1 0 1 1 1 0 0 0 0 1 1 0 1 0 1 1 1 0 1 1 1 1 0 1 1
 1 1 1 0 1 0 0 0 0 0 1 1 1 0 1 0 1 1 1 0 1 0 0 1 0 1 1 1 0 1 0 1 1 0 1 0 0
 1 1 0 1 0 0 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 0 0 1 1 0 1 1 0 1 0 1 1 0 1 1 0
 0 1 1 1 0 0 0 1 0 0 1 0 1 1 1 1 0 0 1 1 1 1 1 1 1 1 0 0 1 1 0 1 1 1 1 1 0
 0 1 1 1 0 1 0 1 0 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 0 1 0 0 0 0 0 1 1
 1 1 0 1 1 1 0 1 1 1 1 1 0 1 1 0 0 0 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 1 1 1 0
 1 1 1 1 1]

print(type(t2))
-> <class 'numpy.ndarray'>
