import os
import numpy as np
import pandas as pd
import cv2
import pyfeats
from scipy.optimize import curve_fit
from scipy import signal
from tqdm import tqdm
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
    
#     eff_df = pd.DataFrame()
#     func_arg_list = [2,3,3,3,2,3,4,2,3,4,5]
#     func_list = [Farrell, Ex, Ga, Lo, ML2, ML3, ML4, MG2, MG3, MG4, GaussL]
#     for i_arg,i_func in zip(func_arg_list,func_list):
        
#         ##### fitting前のプロファイルの処理 #####
#         # 散乱データのlogをとる,正規化する
#         profile = np.exp(intensity)
#         profile_norm = profile/profile.max()

#         # 欠損値補間（線形補間）
#         profile_norm = profile_norm.interpolate(method='linear')

#         eff, cov = curve_fit(i_func,
#                              distance,
#                              profile_norm,
#                              maxfev=20000,
#                              bounds=(tuple([0 for i in range(i_arg)]),
#                                      tuple([np.inf for i in range(i_arg)])
#                                     )
#                             )
        
#         # fittingの精度を算出
#         if i_arg == 2:
            
#             if i_func == Farrell:
                
#                 # 散乱データのlogをとる,正規化する
#                 profile = 10**(intensity)
#                 profile_norm = profile/profile.max()
                
#                 # 欠損値補間（線形補間）
#                 profile_norm = profile_norm.interpolate(method='linear')

#                 eff, cov = curve_fit(i_func,
#                                  distance,
#                                  profile_norm,
#                                  maxfev=20000,
#                                  bounds=(tuple([0 for i in range(i_arg)]),
#                                          tuple([np.inf for i in range(i_arg)])
#                                         )
#                                 )
#                 profile_esti = i_func(distance, eff[0],eff[1])
#                 profile_esti = profile_esti/profile_esti.max()
                
                
#             else:
                
#                 profile = np.exp(intensity)
#                 profile_norm = profile/profile.max()
                
#                 # 欠損値補間（線形補間）
#                 profile_norm = profile_norm.interpolate(method='linear')

#                 eff, cov = curve_fit(i_func,
#                                  distance,
#                                  profile_norm,
#                                  maxfev=20000,
#                                  bounds=(tuple([0 for i in range(i_arg)]),
#                                          tuple([np.inf for i in range(i_arg)])
#                                         )
#                                 )
#                 profile_esti = i_func(distance, eff[0],eff[1])
        
#         elif i_arg == 3:
#             profile_esti = i_func(distance, eff[0],eff[1],eff[2])
#         elif i_arg == 4:
#             profile_esti = i_func(distance, eff[0],eff[1],eff[2],eff[3])
#         else:
#             profile_esti = i_func(distance, eff[0],eff[1],eff[2],eff[3],eff[4])
        
#         r2 = r2_score(profile_norm, profile_esti)
#         # 各fittingの精度を確認したときに表示
#         # print(f"R2 of {str(i_func).split(' ')[1].split('.')[-1]} : {r2:.3f}")
        
#         # dfに格納
#         temp = pd.DataFrame(eff).T
#         temp.columns = [f"{str(i_func).split(' ')[1].split('.')[-1]}_{i+1}" for i in range(i_arg)]
        
#         eff_df = pd.concat([eff_df,temp],axis=1)
        
#     return eff_df

################################################################################################


################################################################################################
def fitting(distance,intensity):
    '''
    ※1サンプルに対する処理
    入力値
        - distance : 等間隔に間引きしたdistance
        - intensity : 等間隔に間引きしたintensity
    
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
    
    ##### fitting #####
    
    eff_df = pd.DataFrame()
    func_arg_list = [2,3,3,3,2,3,4,2,3,4,5]
    func_list = [Farrell, Ex, Ga, Lo, ML2, ML3, ML4, MG2, MG3, MG4, GaussL]
    for i_arg,i_func in zip(func_arg_list,func_list):
        
        ##### fitting前のプロファイルの処理 #####
        # 散乱データのlogをとる,正規化する
        profile = np.exp(intensity)
        profile_norm = profile/profile.max()

        # 欠損値補間（線形補間）
        profile_norm = profile_norm.interpolate(method='linear')

        eff, cov = curve_fit(i_func,
                             distance,
                             profile_norm,
                             maxfev=20000,
                             bounds=(tuple([0 for i in range(i_arg)]),
                                     tuple([np.inf for i in range(i_arg)])
                                    )
                            )
        
        # fittingの精度を算出
        if i_arg == 2:
            
            if i_func == Farrell:
                
                # 散乱データのlogをとる,正規化する
                profile = 10**(intensity)
                profile_norm = profile/profile.max()
                
                # 欠損値補間（線形補間）
                profile_norm = profile_norm.interpolate(method='linear')

                eff, cov = curve_fit(i_func,
                                 distance,
                                 profile_norm,
                                 maxfev=20000,
                                 bounds=(tuple([0 for i in range(i_arg)]),
                                         tuple([np.inf for i in range(i_arg)])
                                        )
                                )
                profile_esti = i_func(distance, eff[0],eff[1])
                profile_esti = profile_esti/profile_esti.max()
                
                
            else:
                
                profile = np.exp(intensity)
                profile_norm = profile/profile.max()
                
                # 欠損値補間（線形補間）
                profile_norm = profile_norm.interpolate(method='linear')

                eff, cov = curve_fit(i_func,
                                 distance,
                                 profile_norm,
                                 maxfev=20000,
                                 bounds=(tuple([0 for i in range(i_arg)]),
                                         tuple([np.inf for i in range(i_arg)])
                                        )
                                )
                profile_esti = i_func(distance, eff[0],eff[1])
        
        elif i_arg == 3:
            profile_esti = i_func(distance, eff[0],eff[1],eff[2])
        elif i_arg == 4:
            profile_esti = i_func(distance, eff[0],eff[1],eff[2],eff[3])
        else:
            profile_esti = i_func(distance, eff[0],eff[1],eff[2],eff[3],eff[4])
        
        r2 = r2_score(profile_norm, profile_esti)
        # 各fittingの精度を確認したときに表示
        # print(f"R2 of {str(i_func).split(' ')[1].split('.')[-1]} : {r2:.3f}")
        
        # dfに格納
        temp = pd.DataFrame(eff).T
        temp.columns = [f"{str(i_func).split(' ')[1].split('.')[-1]}_{i+1}" for i in range(i_arg)]
        
        eff_df = pd.concat([eff_df,temp],axis=1)
        
    return eff_df

################################################################################################
def calc_diff(distance, data_smoothing):
    '''
    - 平滑化済プロファイルの変化率を算出する
    - 1mmごとの変化率を算出
    '''
    # 空の集合
    diff_df =  pd.DataFrame()
    
    # 変化率を算出するプロセス
    for i in range(data_smoothing.shape[1]):
        # 1 mmごとの強度とindexを取得
        index_list = []
        intensity_list = []
        for j in range(29):

            sg_i = data_smoothing.iloc[:,i]
            sg_i.index = distance.index
            
            index_temp  = distance[distance < (j+1)].index.tolist()[-1]
            intensity_temp = np.mean(sg_i[index_temp])

            index_list.append(index_temp)
            intensity_list.append(intensity_temp)

        # プロファイルの強度がnanで得られない箇所は欠損値補間（スプライン補間）
        temp_series = pd.Series(intensity_list,index=range(len(intensity_list)))
        temp_series = temp_series.interpolate('spline',order=2)
        intensity_list = temp_series.values.tolist()

        # 各位置における変化率を算出し、格納
        diff_list = [intensity_list[j+1] - intensity_list[j] for j in range(len(intensity_list)-1)]
        diff_df_temp = pd.DataFrame(diff_list,
                               columns = [data_smoothing.columns[i]],
                               index = [f'diff_{j+1}' for j in range(len(diff_list))])

        diff_df = pd.concat([diff_df, diff_df_temp],axis=1)
        
    return diff_df

################################################################################################
def image2feature(path_folder):
    '''
    paramter
        - path_folder : 画像郡が入っているフォルダのディレクトリ
    return
        - feature (dataframe) : 指定した貯蔵期間・各サンプルごとの画像特徴量
        
    各サンプルのHDR画像から特徴量を抽出
    '''
    
    # pathから画像の名前を取得
    sample_info = os.listdir(path_folder)
    sample_info.sort()
    
    # 各画像に対して以下の処理を実行
    feature_all = pd.DataFrame()
    
    for i_sample in sample_info:
        
        ### 画像情報を読み込む（特徴量算出の前準備）
        img = cv2.imread(f'{path_folder}/{i_sample}',0)
        img = cv2.GaussianBlur(img,(15,15),5)

        # 2値化
        thresh, img_binary = cv2.threshold(img, 0, img.max(), cv2.THRESH_OTSU)

        # mask処理
        mask = np.where(img<thresh, 0, img)
        mask_del_img = np.where(mask>=255,0, mask) # 飽和している場所は0にする処理（ROIから除去）
        mask_del = mask_del_img[mask_del_img>0] # mask_del_imgの0以上の強度のみ算出

        # テクスチャ解析用のmaskを用意 (0,1にする必要がある)
        mask_texture = np.where(mask_del_img>0,1,0)
        
        ### 画像特徴量を順に算出
        
        # First Order Statistics/Statistical Features（16個）
        temp_fos = pyfeats.fos(mask_del_img, mask_texture)
        area = len(mask_del)
        img_std = np.std(mask_del)
        smoothness = 1 - 1/(1+img_std**2)
        
        temp_feature = np.append(temp_fos[0],[area,smoothness])
        temp_label = temp_fos[1]
        temp_label.extend(['FOS_Area','FOS_smoothness'])
        temp_fos = pd.DataFrame(temp_feature,
                            index=temp_label)
        
        # GLCM（14個）
        features_mean, features_range, labels_mean, labels_range = pyfeats.glcm_features(mask_del_img,
                                                                                         ignore_zeros=True)
        temp_glcm = pd.DataFrame(features_mean,index=labels_mean)

        # NGTDM（5個）
        features, labels = pyfeats.ngtdm_features(mask_del_img, mask_texture, d=1)
        temp_ngtdm = pd.DataFrame(features,index=labels)

        # Statistical Feature Matrix (SFM, 4個)
        features, labels = pyfeats.sfm_features(mask_del_img, mask_texture, Lr=4, Lc=4)
        temp_sfm = pd.DataFrame(features,index=labels)

        # Law's Texture Energy Measures (LTE/TEM, 6個)
        features, labels = pyfeats.lte_measures(mask_del_img, mask_texture, l=7)
        temp_lte =  pd.DataFrame(features,index=labels)

        # Fractal Dimension Texture Analysis (FDTA, 4個)
        h, labels = pyfeats.fdta(mask_del_img, mask_texture, s=3)
        temp_fdta =  pd.DataFrame(h,index=labels)

        # Gray Level Run Length Matrix (GLRLM, 11個)
        features, labels = pyfeats.glrlm_features(mask_del_img, mask_texture, Ng=256)
        temp_glrlm =  pd.DataFrame(features,index=labels)

        # Fourier Power Spectrum (FPS, 2個)
        features, labels = pyfeats.fps(mask_del_img, mask_texture)
        temp_fps =  pd.DataFrame(features,index=labels)

        # ↓↓↓↓↓↓↓↓　時間がかかるので，一旦除去
        # # Gray Level Size Zone Matrix (GLSZM, 14個)
        # features, labels = pyfeats.glszm_features(mask_del_img, mask_texture)
        # temp_glszm =  pd.DataFrame(features,index=labels)

        # Local Binary Pattern (LPB, 6個)
        features, labels = pyfeats.lbp_features(mask_del_img, mask_texture, P=[8,16,24], R=[1,2,3])
        temp_lpb =  pd.DataFrame(features,index=labels)
        

        feature_temp = pd.concat([temp_fos, temp_glcm, temp_ngtdm, 
                                  temp_sfm, temp_lte, temp_fdta,
                                  temp_glrlm, temp_fps,# , temp_glszm 
                                  temp_lpb],axis=0).T
        
        feature_all = pd.concat([feature_all, feature_temp],axis=0)
        
    return feature_all

'''
以前までのfittingのやつ（現在はfitting関数に統一済み）
コメントアウトを解除すれば、利用可能
'''

# ################################################################################################
# def fitting_farrell_1(distance, intensity):
#     '''
#     ※1サンプルに対する処理
#     ※これらの関数は規格化はしなくて良い
#     入力値
#         - distance : 等間隔に間引きしたdistance
#         - intensity : 等間隔に間引きしたintensity
    
#     以下の関数をfitting
#         - Farrell関数
#             - 1. 吸収係数・等価散乱係数に対するfitting
    
#     返り値：[u_s, u_a], df
#     '''
#     ##### Farrell関数① #####
    
#     # 関数の定義
#     def farrell_eq(x, u_s, u_a):
#         a_prime = u_s/(u_a+u_s)
#         u_eff = np.sqrt(abs(3*u_a*u_s))
#         u_t = u_a +u_s
#         n_r = 1.35
#         r_d = -1.44*(1/n_r**2) + 0.710*(1/n_r) + 0.668 + 0.063*n_r
#         A = (1+r_d)/(1-r_d)
#         r_1 = np.sqrt((1/u_t)**2 + x**2)
#         r_2 = np.sqrt((1/u_t + 4*A/(3*u_t))**2 + x**2)

#         return (a_prime/(4*np.pi))*((1/u_t)*(u_eff+1/r_1)*np.exp(-1*u_eff*r_1)/(r_1**2)) + ((1/u_t + 4*A/(3*u_t))*(u_eff+1/r_2)*np.exp(-1*u_eff*r_2)/(r_2**2))
    
#     # 最小二乗法によるパラメータ推定
#     rscore_list = []
#     mse_list = []
#     index_name = []

#     # 散乱データのlogをとる,正規化する
#     profile = np.exp(intensity)
#     profile_norm = profile/profile.max()
    
#     # 欠損値補間（線形補間）
#     profile_norm = profile_norm.interpolate(method='linear')

    
#     # 先行研究では，u_s = 0.100 ~ 2.00
#     # 先行研究では，u_a = 0.0100 ~ 0.0500
#     # 初期値設定
    
#     u_s = 0.100
#     u_a = 0.001

#     for i in range(100):
#         for j in range(100):

#             profile_temp = farrell_eq(distance,u_s,u_a)
#             profile_temp = profile_temp/profile_temp.iloc[0]

#             index_name.append([f'u_s-{u_s:.6f}-u_a-{u_a:.6f}'])
#             rscore_list.append(r2_score(profile_norm, profile_temp))
#             mse_list.append(((profile_norm - profile_temp)**2).sum())

#             u_a += 4*1e-4
#         u_s += 1.9*1e-2
#             # これらの値は上述の範囲に合わせて，当てはめた

#     # MSEを最小とするパラメータの選択
#     df = pd.DataFrame([index_name,rscore_list,mse_list]).T
#     df.columns = ['optical','r2','MSE']
#     df = df.sort_values(by='MSE',ascending=True)
#     eff = [float(df.iloc[0,0][0].split('-')[1]),float(df.iloc[0,0][0].split('-')[3])]
    
#     return eff, df
    
# ################################################################################################
# def fitting_farrell_2(distance, intensity):
#     '''
#     ＊＊＊＊このfittingは使えない．．．．残念＊＊＊＊
#     ※1サンプルに対する処理
#     入力値
#         - distance : 等間隔に間引きしたdistance
#         - intensity : 等間隔に間引きしたintensity
    
#     以下の関数をfitting
#         - Farrell関数 (計4つ)
#             - 2. a,b　（両係数の積)に関するfitting
    
#     返り値 a, b
#     '''
#     ##### Farrell関数② #####
#     # 関数の定義
#     # fittingする関数を定義
#     def func_farrell(x,a,b):
#         y = -2*np.log(x) -a*x + b
#         return y
    
#     # 欠損値補間（線形補間）
#     intensity = intensity.interpolate(method='linear')
#     profile_norm = intensity
    
#     # fitting処理
#     eff,cov = curve_fit(func_farrell,
#                         distance,
#                         profile_norm,
#                         maxfev=20000)
#     # fittingの精度を算出
#     profile_esti = func_farrell(distance, eff[0],eff[1])
#     r2 = r2_score(profile_norm, profile_esti)
    
#     return eff, r2

# ################################################################################################
# def ML(distance, intensity):
#     '''
#     ※1サンプルに対する処理
#     入力値
#         - distance : 等間隔に間引きしたdistance
#         - intensity : 等間隔に間引きしたintensity
    
#     以下の関数をfitting
#         - Modified-Lorentzian関数
        
#     返り値 a, b, c, d
#     '''
#     ##### Modified-Lorentzian関数 #####
#     def lorenz(x,a,b,c,d):
#         '''
#         ローレンツ曲線を少し変形
#         '''
#         return a + b/(c+x**d)
    
#     # 散乱データのlogをとる,正規化する
#     profile = np.exp(intensity)
#     profile_norm = profile/profile.max()
    
#     # 欠損値補間（線形補間）
#     profile_norm = profile_norm.interpolate(method='linear')
    
#     eff,cov = curve_fit(lorenz,
#                     distance,
#                     profile_norm,
#                     check_finite=False,
#                     maxfev=20000,
#                     bounds=((0,0,0,0),(np.inf,np.inf,np.inf,np.inf)))
    
#     # fittingの精度を算出
#     profile_esti = lorenz(distance, eff[0],eff[1],eff[2],eff[3])
#     r2 = r2_score(profile_norm, profile_esti)
    
#     return eff, r2

# ################################################################################################
# def MG(distance, intensity):
#     '''
#     ※1サンプルに対する処理
#     入力値
#         - distance : 等間隔に間引きしたdistance
#         - intensity : 等間隔に間引きしたintensity
    
#     以下の関数をfitting
#         - Modified-Gompertz関数
        
#     返り値 a, b, c, d
#     '''
#     ##### Modified-Gompertz関数 #####
#     def Gompertz(x,a,b,c,d):
#         '''
#         ゴンペルツ関数
#         '''
#         return a -b*np.exp(-np.exp(c-d*x))
    
#     # 散乱データのlogをとる,正規化する
#     profile = np.exp(intensity)
#     profile_norm = profile/profile.max()
    
#     # 欠損値補間（線形補間）
#     profile_norm = profile_norm.interpolate(method='linear')
    
    
#     eff,cov = curve_fit(Gompertz,
#                     distance,
#                     profile_norm,
#                     check_finite=False,
#                     maxfev=20000,
#                     bounds=((0,0,0,0),(np.inf,np.inf,np.inf,np.inf)))
    
#     # fittingの精度を算出
#     profile_esti = Gompertz(distance, eff[0],eff[1],eff[2],eff[3])
#     r2 = r2_score(profile_norm, profile_esti)
    
#     return eff, r2

# ################################################################################################
# def GL(distance, intensity):
#     '''
#     ※1サンプルに対する処理
#     入力値
#         - distance : 等間隔に間引きしたdistance
#         - intensity : 等間隔に間引きしたintensity
    
#     以下の関数をfitting
#         - Gaussian-Lorentizian関数
        
#     返り値 a, b, c, d, e
#     '''
#     ##### Gaussian-Lorentizian関数 #####
#     def GaussL(x,a,b,c,d,e):
#         '''
#         ガウシアン-ローレンツ関数
#         '''
#         return a + b/(1+e*((x-c)/d)**2)*np.exp((((1-e)/2)*((x-e)/d))**2)
    
#     # 散乱データのlogをとる,正規化する
#     profile = np.exp(intensity)
#     profile_norm = profile/profile.max()
    
#     # 欠損値補間（線形補間）
#     profile_norm = profile_norm.interpolate(method='linear')
    
#     eff,cov = curve_fit(GaussL,
#                     distance,
#                     profile_norm,
#                     check_finite=False,
#                     maxfev=20000,
#                     bounds=((0,0,0,0,0),(np.inf,np.inf,np.inf,np.inf,np.inf)))
    
#     # fittingの精度を算出
#     profile_esti = GaussL(distance, eff[0],eff[1],eff[2],eff[3],eff[4])
#     r2 = r2_score(profile_norm, profile_esti)
    
#     return eff, r2


        
        
        
        
        
        


    
        
    
