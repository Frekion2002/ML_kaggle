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
- 결측치 보간 (`0 대체`, `최댓값 기반 보정`)
- 총량/평균 기반 **특성 공학**
- **주기성 반영**: 날짜 → `sin/cos` 변환
- **Magnus 공식**으로 기온 역산 feature 추가

### 2. Feature Importance (SHAP 기준)

```
1. 평균 기온 (1.38)
2. 23시 기온 (0.74)
3. sin/cos 날짜 변환 (0.29 / 0.20)
...
```

<SHAP 그래프 이미지가 있다면 여기에 첨부>

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
- **5-Fold Cross Validation**:  
  → 평균 RMSE: `1.2916 ± 0.0268`

---

## 👥 Team Contribution

| 팀원 | 주요 역할 |
|------|-----------|
| 곽용진 | 모델 구현, Optuna 튜닝, SHAP 시각화, 과적합 진단 |
| 은채웅 | 데이터 전처리, 피처 공학, 모델 비교 분석, 차원 축소 |

협업은 `GitHub`, `정기 회의`, `역할 분담`을 통해 진행했습니다.

---

## 🚧 Limitations & Future Work

### ❌ 한계점
- 극단적 기후 현상 예측 어려움
- 지역 특수성 반영 한계
- 장기 예측 부적합

### ✅ 향후 연구 방향
- **딥러닝 통합** (LSTM 등 시계열 모델)
- **지역 특화 모델 개발**
- **위성/레이더 데이터 연계**

---

## 🛠️ Usage

```bash
# 필수 라이브러리 설치
pip install -r requirements.txt

# 모델 학습 및 예측 실행
python train.py
```

- `data/`: 기상 데이터 CSV
- `models/`: 학습된 모델 저장 경로
- `notebooks/`: 분석 및 시각화 노트북

---

## 📚 References

- [Kaggle Challenge Dataset](https://www.kaggle.com/competitions/next-day-air-temperature-forecast-challenge-2/data)
- [Magnus 공식 설명](https://www.ecmwf.int/sites/default/files/elibrary/2015/17326-skill-ecmwf-cloudiness-forecasts.pdf)
- [Optuna 튜토리얼](https://optuna.readthedocs.io/en/stable/)
- [XGBoost 파라미터](https://xgboost.readthedocs.io/en/release_3.0.0/parameter.html)
- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

---

## 📌 License

This project is for academic purposes only.
