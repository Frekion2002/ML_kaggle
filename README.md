# ML_kaggle
---
# 🌤️ Next-Day Temperature Forecast using Machine Learning
---

## 📌 Project Overview

본 프로젝트는 머신러닝 기법을 활용하여 **다음날 평균 기온을 예측**하는 모델을 개발합니다.  
국가기상청의 2019–2024년 데이터를 활용하여 다양한 기상 변수의 전처리 및 특성 공학을 수행하고,  
`XGBoost`, `Random Forest`, `SVM` 등의 모델 성능을 비교했습니다.

> 🎯 최종 결과:  
> `XGBoost 앙상블` 모델이 **RMSE 1.2440°C**로 최우수 성능을 달성했습니다.

---

## 🧠 ML Pipeline

### 1. Data Preprocessing
- 결측치 보간
  0으로 대체 : 일조량, 적설, 강수량, 시정, 풍속, 증하층운량 → '측정값 없음' or '해당 현상 없음'
  최댓값 기반 보정 : 최저운고 → "관측할 구름이 없기 때문에"
- 총량/평균 기반 **특성 공학**
- **주기성 반영**: 날짜 → `sin/cos` 변환 (1월 1일과 12월 31일은 숫자상 멀지만 날씨는 유사함을 표현하기 위해)
- **Magnus 공식**으로 기온 역산 feature 추가 (의미 있는 feature 추가)

### 2. Feature Importance (SHAP 기준)

```
1. 평균 기온 (1.38)
2. 23시 기온(Magnus 역산으로 추정) (0.74)
3. sin/cos 날짜 변환 (0.29, 0.2)
4. 이슬점 (0.21)
...
```

![image](https://github.com/user-attachments/assets/cf713bbf-f7e0-45fa-9ae3-7dac7add0f7d)

---

## 🤖 Model Comparison

| 모델 | RMSE (°C) | 특징 |
|------|-----------|------|
| 🥇 XGBoost 앙상블 | **1.2440** | 높은 정확도, 결측치에 강함 |
| Random Forest | 1.4120 | 특성 중요도 제공 |
| SVM | 2.8543 | 고차원에 강하지만 느림 |

- **튜닝 방법**: Optuna (`Bayesian Optimization`)
- **앙상블 기법**: Soft Voting with 10 random seeds

---

## 🧪 Evaluation

- **Metric**: Root Mean Squared Error (RMSE)
  → 직관적 해석, 대오차 강조, 산업 표준기반
- **5-Fold Cross Validation**:  
  → 평균 RMSE: `1.2916 ± 0.0268`

---

## 🧠 고찰
- 머신러닝 속 feature engineering이 모델 성능에 큰 영향을 미친다는 것을 확인
- 도메인 데이터에 대한 이해가 머신러닝에 대한 이해 못지 않게 중요하다는 것을 깨닳음
- 극단적인 상황에 대한 예측 성능 보완을 위해 추가적인 외부 변수 도입이 필요

## 📚 References

- [Kaggle Challenge Dataset](https://www.kaggle.com/competitions/next-day-air-temperature-forecast-challenge-2/data)
- [Magnus 공식 설명](https://www.ecmwf.int/sites/default/files/elibrary/2015/17326-skill-ecmwf-cloudiness-forecasts.pdf)
- [Optuna 튜토리얼](https://optuna.readthedocs.io/en/stable/)
- [XGBoost 파라미터](https://xgboost.readthedocs.io/en/release_3.0.0/parameter.html)
- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

---

## More information

- Github에 업로드 된 PDF 참고
