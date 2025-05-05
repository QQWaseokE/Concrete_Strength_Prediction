import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

## 데이터 분석
df = pd.read_csv("Concrete_Compressive_Strength_dataset.csv")
print(df.describe())
print(df.isnull().sum())

sbn.boxplot(data=df) # boxplot으로 이상치 시각화
plt.show()

## 데이터 전처리
# IQR방식으로 제거 (IQR = 데이터 중간 50% 범위)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5*IQR)) | (df > (Q3 + 1.5*IQR))).any(axis=1)]

# 전처리 후 데이터 확인
print(df.describe())
sbn.boxplot(data=df)
plt.show()

# 파생 변수 생성
df["W/C_Ratio"] = df["Water"] / df["Cement"]

## 데이터 관계 분석
sbn.heatmap(df.corr(), annot=True) # Concrete_Strength와 상관관계 높은 변수 찾기
plt.show()

# 산점도 시각화
sbn.scatterplot(x="Cement", y="Concrete_Strength", data=df)
plt.show()

sbn.scatterplot(x="W/C_Ratio", y="Concrete_Strength", data=df)
plt.show()

## 모델링
# 1. 피처(X)와 타겟(y) 분리
X = df.drop("Concrete_Strength", axis=1)  # 타겟 제외 모든 컬럼
y = df["Concrete_Strength"]              # 타겟 변수

# 2. 훈련/테스트 세트 분할 (7:3 비율)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 모델 훈련
model = RandomForestRegressor(random_state=42)  # 재현성을 위해 random_state 설정
model.fit(X_train, y_train)

# 4. 평가
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)**0.5
print(f"RMSE: {rmse:.2f} MPa")

# 5. 변수 중요도 시각화
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

feature_importance.plot.barh(x='Feature', y='Importance', figsize=(10,6))
plt.title("RandomForest Feature Importance")
plt.show()

# Age vs Strength 관계 확인  
sbn.regplot(x="Age_Day", y="Concrete_Strength", data=df, order=2)  # 2차 다항식 피팅  
plt.show()