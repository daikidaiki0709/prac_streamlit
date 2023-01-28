import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import signal
from sklearn.metrics import r2_score
import streamlit as st

################################################################################################
def delete_distance(distance, apple_df):
    '''
    最大30 mmまでのdataframeを作成
    '''
    
    # 1 mm未満を削除する処理
    apple_df = apple_df.iloc[distance.index,:]
    distance.index = [i for i in range(distance.shape[0])]
    apple_df.index = [i for i in range(apple_df.shape[0])]

    # 最大30 mmとした処理
    distance_30mm = distance[distance <= 30.1]
    apple_df_30mm = apple_df.iloc[distance_30mm.index,:]
    
    return distance_30mm, apple_df_30mm

################################################################################################
def skip_distance(distance, apple_df):
    '''
    0.5 mm ~ 20 mmのdistance範囲が解析範囲
    1 mm間隔となるindexを返す
    '''
    apple_df.index = distance.index
    index_num = []
    for i in range(29):
        index_now = i
        temp = apple_df[(distance>=index_now)&(distance<=(index_now+1))]
        index_num.append(temp.index[-1])

    distance_eq = distance.loc[index_num]
    apple_df_eq = apple_df.loc[index_num,:]
    
    return distance_eq, apple_df_eq

################################################################################################
def correct_intensity(distance,apple_df):
    '''
    散乱強度を補正する関数
    果実面をランバート面と仮定
    参考：Principles and Applications of Light Backscattering Imagingin Quality Evaluation of Agro-food Products: a Review
    '''
    
    S = 80 # りんごの径
    
    profile_df = pd.DataFrame()
    for i in range(apple_df.shape[1]):
        i_profile = apple_df.iloc[:,i]
        temp = i_profile*S/np.sqrt(S**2 - distance**2)
        profile_df[f'{apple_df.columns[i]}'] = temp
        
    return profile_df
################################################################################################
def correct_distance(distance):
    '''
    散乱距離を補正する関数
    果実を球と仮定
    参考　：　Improving apple fruit firmness predictions by effective correction of multispectral scattering images.
    '''
    
    S = 80 # りんごの径
    
    distance_df = pd.DataFrame()# 補正後の距離を格納
    temp_list = []
    
    for i in range(distance.shape[0]):
        i_distance = distance[i]
        temp = S*np.arctan(i_distance/np.sqrt(S**2 - i_distance**2))
        temp_list.append(temp)
    
    distance_df = pd.Series(temp_list,name='distance (mm)')
        
    return distance_df

################################################################################################
def smoothing_apple(distance, data):
    '''
    平滑化
        - 生プロファイルから0.062 mm間隔で取得
          （その際にノイズ除去のため，取得点の前後で平滑化（平均化））
        - 抽出したプロファイルを窓サイズ3で平滑化（ノイズを減らす）

    平滑化の条件は以下
        - 手法：Savitky-Golay法
        - 窓サイズ：3
        - 次元：1
    ''' 
    
    # 生プロファイルから 0.062 mm間隔で取得
    index_num = []
    width = 0.062
    for index_i in range(int(30/width)+1):
        temp = data[(distance>=width*index_i)&(distance<=(width*(index_i+1)))]
        index_num.append(temp.index[0])
    distance_eq = distance.loc[index_num]
    apple_df_eq = data.loc[index_num,:]
    
    # infをnanに変更
    apple_df_eq = apple_df_eq.replace([-np.inf,np.inf],np.nan)
    
    # 欠損値の補間
    apple_df_eq = apple_df_eq.interpolate(method='index')
    apple_df_eq = apple_df_eq.interpolate('ffill')
    apple_df_eq = apple_df_eq.interpolate('bfill')

    # 空の集合
    sg_total = pd.DataFrame()

    # 同時に行いそれぞれに格納
    for i in range(data.shape[1]):

        # 平滑化処理
        sg_i = pd.Series(signal.savgol_filter(apple_df_eq.iloc[:,i],3,1))

        #平滑化処理の結果を格納
        sg_total = pd.concat([sg_total,sg_i],axis=1)

    # 平滑化後のプロファイルを整理
    sg_total.columns = data.columns
    
    return distance_eq, sg_total

################################################################################################
def fitting_func(distance,intensity,func_name):
    '''
    ※1サンプルに対する処理
    入力値
        - distance : 等間隔に間引きしたdistance
        - intensity : 等間隔に間引きしたintensity
        - func_name : fittingする関数の名前
    
    以下の関数をfitting (計 34　paramters)
        - Farrell (2 parameters)
        - Exponential (3 parameters)
        - Gaussian (3 paramters)
        - Lorentzian (3 paraemters)
        - Modified Lorentzian 2 (2 paramters)
        - Modified Lorentzian 3 (3 paramters)
        - Modified Lorentzian 4 (4 paramters)
        - Modified Gompertz 2 (2 paramters)
        - Modified Gompertz 3 (3 paramters)
        - Modified Gompertz 4 (4 paramters)
        - Gaussian Lorentzian (5 paramters)
    
    返り値
        - 各パラメータのdataframe
    '''
    ##### 関数の作成 #####
    def Farrell(x,a,b):
        '''
        Farrell式
        プロファイルを指数乗する
         -> np.exp(profile)
        '''
        return (a/(x**2))*np.exp(-b*x)

    def Ex(x,a,b,c):
        '''
        指数関数 (exponential)
        '''
        return a + b*np.exp(-x/c)

    def Ga(x,a,b,c):
        '''
        ガウス関数 (Gaussian)
        '''
        return a + b*np.exp(-0.5*(x/c)**2)

    def Lo(x,a,b,c):
        '''
        ローレンツ関数 (Lorentzian)
        '''
        return a + b/(1+(x/c)**2)

    def ML2(x,a,b):
        '''
        修正ローレンツ関数2 (Modified-Lorentzian 2)
        '''
        return 1/(1+(x/a)**b)

    def ML3(x,a,b,c):
        '''
        修正ローレンツ関数3 (Modified-Lorentzian 3)
        '''
        return a + (1-a)/(1+(x/b)**c)

    def ML4(x,a,b,c,d):
        '''
        修正ローレンツ関数4 (Modified-Lorentzian 3)
        '''
        return a + b/(c+x**d)

    def MG2(x,a,b):
        '''
        修正ゴンペルツ関数2 (Modified-Gompertz 2)
        '''
        return 1 - np.exp(-np.exp(a - b*x))

    def MG3(x,a,b,c):
        '''
        修正ゴンペルツ関数3 (Modified-Gompertz 3)
        '''
        return 1 - (1-a)*np.exp(-np.exp(b - c*x))

    def MG4(x,a,b,c,d):
        '''
        修正ゴンペルツ関数4 (Modified-Gompertz 4)
        '''
        return a - (b*np.exp(-np.exp(c - d*x)))
    
    def GaussL(x,a,b,c,d,e):
        '''
        Gaussian-Lorentizian関数
        '''
        return a + b/(1+e*((x-c)/d)**2)*np.exp((((1-e)/2)*((x-e)/d))**2)
    
    
    # 条件
    if func_name == 'Farrell':
        func = Farrell
    elif func_name == 'Ex':
        func = Ex
    elif func_name == 'Ga':
        func = Ga
    elif func_name == 'Lo':
        func = Lo
    elif func_name == 'ML2':
        func = ML2
    elif func_name == 'ML3':
        func = ML3
    elif func_name == 'ML4':
        func = ML4
    elif func_name == 'MG2':
        func = MG2
    elif func_name == 'MG3':
        func = MG3
    elif func_name == 'MG4':
        func = MG4
    else:
        func = GaussL
    
    ##### fitting #####
    
    if func in [Farrell,ML2,MG2]:
        
        if func == Farrell: 
            # 散乱データのlogをとる,正規化する
            profile = 10**(intensity)
            profile_norm = profile/profile.max()

            # 欠損値補間（線形補間）
            profile_norm = profile_norm.interpolate(method='linear')

            eff, cov = curve_fit(func,
                             distance,
                             profile_norm,
                             maxfev=20000,
                             bounds=(tuple([0 for i in range(2)]),
                                     tuple([np.inf for i in range(2)])
                                    )
                            )
            profile_esti = func(distance, eff[0],eff[1])
            profile_esti = profile_esti/profile_esti.max()
            
        else:
            profile = np.exp(intensity)
            profile_norm = profile/profile.max()

            # 欠損値補間（線形補間）
            profile_norm = profile_norm.interpolate(method='linear')

            eff, cov = curve_fit(func,
                             distance,
                             profile_norm,
                             maxfev=20000,
                             bounds=(tuple([0 for i in range(2)]),
                                     tuple([np.inf for i in range(2)])
                                    )
                            )
            profile_esti = func(distance, eff[0],eff[1])
            
        
    elif func in [Ex, Ga, Lo, ML3, MG3]:
        
        profile = np.exp(intensity)
        profile_norm = profile/profile.max()

        # 欠損値補間（線形補間）
        profile_norm = profile_norm.interpolate(method='linear')

        eff, cov = curve_fit(func,
                         distance,
                         profile_norm,
                         maxfev=20000,
                         bounds=(tuple([0 for i in range(3)]),
                                 tuple([np.inf for i in range(3)])
                                )
                        )
        profile_esti = func(distance, eff[0],eff[1],eff[2])
        
    elif func in [ML4,MG4]:
        profile = np.exp(intensity)
        profile_norm = profile/profile.max()

        # 欠損値補間（線形補間）
        profile_norm = profile_norm.interpolate(method='linear')

        eff, cov = curve_fit(func,
                         distance,
                         profile_norm,
                         maxfev=20000,
                         bounds=(tuple([0 for i in range(4)]),
                                 tuple([np.inf for i in range(4)])
                                )
                        )
        profile_esti = func(distance, eff[0],eff[1],eff[2],eff[3])

    else:
        profile = np.exp(intensity)
        profile_norm = profile/profile.max()

        # 欠損値補間（線形補間）
        profile_norm = profile_norm.interpolate(method='linear')

        eff, cov = curve_fit(func,
                         distance,
                         profile_norm,
                         maxfev=20000,
                         bounds=(tuple([0 for i in range(5)]),
                                 tuple([np.inf for i in range(5)])
                                )
                        )
        profile_esti = func(distance, eff[0],eff[1],eff[2],eff[3],eff[4])
    
    # 決定係数の算出
    r2 = r2_score(profile_norm, profile_esti)
    
    return profile_esti, r2
    
