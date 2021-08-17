# [1, np.nan, np.nan, 8, 10]
### 결측치 처리(interpolate 사용) - 앞에꺼와 뒤에꺼의 중간값으로 알아서 넣어짐
# 1. 행 삭제
# 2. [1, 0, 0, 8, 10]  0으로 채우기
# 3. [1, 1, 1, 8, 10]  앞에껄로 채우기
# 4. [1, 8, 8, 8, 10]  뒤에껄로 채우기
# 5. [1, 4.5, 4.5, 8, 10]  중위값으로 채우기
# 6. 보간 - interpolate 사용(기준은 linear이다!!!!!!!)
# 7. 모델링 - predict
# 8. 부스트계열(트리계열)은 결측치에 대해 자유(?)롭다.  -  트리계열은 결측치 처리를 안 해줘도 문제가 없다(해주면 좋긴 함)
#   => 행삭제, 보간법 순으로 많이 씀(행 삭제를 제일 많이 씀, 나머지는 조작이라서)
### interpolate : linear을 기준으로 결측치를 채움
from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd
# datetime 형식으로 리스트 만들기
datastrs = ['8/13/2021', '8/14/2021', '8/15/2021', '8/16/2021', '8/17/2021']
dates = pd.to_datetime(datastrs)
print(dates)
print(type(dates))      # <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
print('===================================')
# [1, np.nan, np.nan, 8, 10]와 dates를  시리즈로 변환
ts = Series([1, np.nan, np.nan, 8, 10], index=dates)
print(ts)
### 결측치 채움!(linear기준)
ts_intp_linear = ts.interpolate()       # interpolate : linear기준으로 결측치 채워줌
print(ts_intp_linear)
'''
===================================
2021-08-13     1.0
2021-08-14     NaN  ***
2021-08-15     NaN  ***
2021-08-16     8.0
2021-08-17    10.0
dtype: float64
2021-08-13     1.000000
2021-08-14     3.333333  ***
2021-08-15     5.666667  ***
2021-08-16     8.000000
2021-08-17    10.000000
dtype: float64
'''
