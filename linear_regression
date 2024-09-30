import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':

    # 테스트용 데이터 생성
    x = np.random.rand(1000)*100
    y = 0.8*x+np.random.randn(1000)*30

    # Linear Regrssion model 생성
    model = LinearRegression()

    # Linear Regression model 학습
    model.fit(x.reshape(-1,1), y)

    # Prediction
    y_new = model.predict(np.array([6]).reshape((-1, 1)))
    print("Data Prediction: ", y_new)

    # Linear Regression model 평가
    r_sq = model.score(x.reshape(-1,1), y)
    print("결정 계수: ", r_sq)

    # Linear Model 식
    b0,b1 = model.intercept_, model.coef_[0]
    print("b0",b0)
    print("b1",b1)

    # 시각화
    plt.scatter(x, y, s=5)
    plt.plot(x, model.predict(x.reshape(-1,1)), color='red', linewidth=2)
    plt.annotate('y = '+str(round(b1,2))+'x '+str(round(b0,2)), xy=(100, 100), xytext=(80, 80))
    plt.show()