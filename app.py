import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from FuncLaserScattering import delete_distance, skip_distance, correct_distance, correct_intensity, smoothing_apple, fitting_func
import japanize_matplotlib

# st.set_page_config(layout="wide")

st.title("デモアプリ")

st.subheader('レーザー散乱法で取得したプロファイルへfittingさせる関数を選択する')

storage_menu = st.text_input('貯蔵期間を入力してください（00,01,02,03,04のどれか）','00')
sample_menu = int(st.text_input('サンプル番号を入力してください（0〜39のどれか）','0'))
wavelength_menu = st.selectbox("どちらの波長のデータを用いますか",["633nm","850nm"])
func_menu = st.selectbox("fittingさせる関数を選択してください", 
                         [
                             "Farrell式（簡易）","指数型関数","ガウス関数",
                             "修正済みローレンツ関数_係数2つ","修正済みローレンツ関数_係数3つ","修正済みローレンツ関数_係数4つ",
                             "修正済みゴンペルツ関数_係数2つ","修正済みゴンペルツ関数_係数3つ","修正済みゴンペルツ関数_係数4つ",
                             "ガウシアン-ローレンツ関数"
                         ]
                        )

##########################################################################################
total_df = pd.read_csv(f'Profile_FirstStorage/{wavelength_menu}/Profile_FirstStorage_{storage_menu}_{wavelength_menu}.csv')

# 距離と輝度（強度）を分割
distance = total_df.iloc[:,0]
apple_df = total_df.drop('distance (mm)',axis=1)


#####解析範囲外の領域を削除#####
distance_30mm, apple_df_30mm = delete_distance(distance, apple_df)

#####散乱距離の補正#####
distance_30mm = correct_distance(distance_30mm)

#####散乱強度の補正#####
apple_df_30mm = correct_intensity(distance_30mm,apple_df_30mm)

#####プロファイルの平滑化#####
distance_smooth, apple_smooth = smoothing_apple(distance_30mm, apple_df_30mm)

#####プロファイルの間引き (1mm間隔)#####
distance_eq, apple_smooth_eq = skip_distance(distance_smooth, apple_smooth)

##########################################################################################


st.subheader('プロファイルとfitting結果を表示')

intensity = apple_smooth_eq.iloc[:,sample_menu]

if func_menu=='Farrell式（簡易）':
    func = 'Farrell'
    inte = 10**intensity
    inte = inte/inte.max()
elif func_menu=="指数型関数":
    func = 'Ex'
    inte = np.exp(intensity)
    inte = inte/inte.max()
elif func_menu=="ガウス関数":
    func = 'Ga'
    inte = np.exp(intensity)
    inte = inte/inte.max()
elif func_menu=="修正済みローレンツ関数_係数2つ":
    func = 'ML2'
    inte = np.exp(intensity)
    inte = inte/inte.max()
elif func_menu=="修正済みローレンツ関数_係数3つ":
    func = 'ML3'
    inte = np.exp(intensity)
    inte = inte/inte.max()
elif func_menu=="修正済みローレンツ関数_係数4つ":
    func = 'ML4'
    inte = np.exp(intensity)
    inte = inte/inte.max()
elif func_menu=="修正済みゴンペルツ関数_係数2つ":
    func = 'MG2'
    inte = np.exp(intensity)
    inte = inte/inte.max()
elif func_menu=="修正済みゴンペルツ関数_係数3つ":
    func = 'MG3'
    inte = np.exp(intensity)
    inte = inte/inte.max()
elif func_menu=="修正済みゴンペルツ関数_係数4つ":
    func = 'MG4'
    inte = np.exp(intensity)
    inte = inte/inte.max()
else:
    func = 'GaussL'
    inte = np.exp(intensity)
    inte = inte/inte.max()

#####fitting#####


esti,r2 = fitting_func(distance_eq,intensity ,func_name=func)


fig = plt.figure(figsize=(5,5))
plt.plot(distance_eq,inte,alpha=1,color='k',label='測定結果')
plt.plot(distance_eq,esti,color='r',label='近似結果')
plt.xlim(0,30)
plt.ylim(0,1)
plt.title(f'{func_menu}｜決定係数：{r2:.3f}')
plt.legend(loc='upper right',frameon=False)
plt.xlabel('入射点からの距離[mm]')
plt.ylabel('後方散乱光の相対強度')


st.pyplot(fig)
