import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import FinanceDataReader as fdr
import mplfinance as mpf
from datetime import datetime, timedelta
import json
import yaml
import streamlit_authenticator as stauth
import numpy as np
import time
import requests as rq
import pymysql
from streamlit_authenticator.utilities.hasher import Hasher
import os.path
import pickle as pkle
from streamlit_js_eval import streamlit_js_eval
from passlib.context import CryptContext
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi

from streamlit_authenticator.utilities import (CredentialsError,
                                               ForgotError,
                                               Hasher,
                                               LoginError,
                                               RegisterError,
                                               ResetError,
                                               UpdateError)


st.set_page_config(
        page_title="ESG 정보 제공 플랫폼",
        page_icon=":earth_africa:",
        layout="centered",
    )
st.title('Kwargs')

# 가상의 데이터셋 생성
years = list(range(2019, 2024))
companies = ['삼성전자', 'SK 하이닉스', 'NAVER', 'LG전자', '현대모비스']
sectors = ['전자', '에너지', '정보기술', '전자', '차량']

data = []
for year in years:
    for company, sector in zip(companies, sectors):
        esg_score = np.random.rand() * 100  # 가상의 ESG 점수
        data.append([company, sector, year, esg_score])

df = pd.DataFrame(data, columns=['company', 'sector', 'year', 'score'])
# 데이터셋 피벗팅
pivot_df = df.pivot(index='company', columns='year', values='score').fillna(0)

model = NearestNeighbors(n_neighbors=3, algorithm='auto')
model.fit(pivot_df)


def recommend_stocks(user_preferences, n_recommendations=3):
    # 유저의 선호도를 기반으로 가상의 ESG 점수를 생성
    user_profile = pd.DataFrame([user_preferences], columns=pivot_df.columns).fillna(0)
    distances, indices = model.kneighbors(user_profile, n_neighbors=n_recommendations)
    
    recommended_companies = pivot_df.index[indices.flatten()].tolist()
    return recommended_companies

# 관심도 저장 및 로드 함수
def save_user_preferences(username, preferences):
    uri = "mongodb+srv://pinku03260707:6C6X9TbrQFpdyqEf@kwargs.za1zguv.mongodb.net/?retryWrites=true&w=majority&appName=kwargs"
    client = MongoClient(uri, server_api=ServerApi('1'))
    doc = {
        'username' : username,
         '전자': 0, '에너지': 0, '헬스케어': 0, '차량': 0, '소재': 0, '화학': 0, '미디어': 0, '건설': 0, '금융': 0, '정보기술': 0, '생활소비재': 0, '운송': 0, '중공업': 0, '유통': 0, 'E': 0, 'S': 0, 'G': 0
    }
    
    for category, score in preferences.items():
        cursor.execute('''
            INSERT INTO preferences (username, category, score) 
            VALUES (%s, %s, %s) 
            ON DUPLICATE KEY UPDATE score=%s
        ''', (username, category, score, score))
       
    
    db.users.insert_one(doc)

def load_user_preferences(username):
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute('SELECT category, score FROM preferences WHERE username=%s', (username,))
    preferences = {category: score for category, score in cursor.fetchall()}
    connection.close()
    return preferences


# 한국거래소 코스피 인덱스에 해당하는 종목 가져오기
@st.cache_data
def getSymbols(market='KOSPI',sort='Marcap'): # 정렬하는 기준을 시장가치(Marcap)으로 함
    df = fdr.StockListing(market)
    # 정렬 설정 (= 시가총액 기준으로는 역정렬)
    ascending = False if sort == 'Marcap' else True
    df.sort_values(by=[sort],ascending=ascending, inplace=True)
    return df[['Code','Name','Market']]

@st.cache_data
def load_stock_data(code, ndays):
    end_date = pd.to_datetime('today')
    start_date = end_date - pd.Timedelta(days=ndays)
    data = fdr.DataReader(code, start_date, end_date)
    return data

# 로그아웃에 대한 함수
def forceLogout():
    authenticator.cookie_manager.delete(authenticator.cookie_name)
    st.session_state['logout'] = True
    st.session_state['name'] = None
    st.session_state['username'] = None
    st.session_state['authentication_status'] = None
    del st.session_state['clickedReset']

# 리셋 버튼이 클릭되었는지에 대해서 그 상태를 반환하는 함수 
def hasClickedReset():
    if 'clickedReset' in st.session_state.keys() and st.session_state['clickedReset']:
        return True
    else:
        return False

# 캔들차트 출력 함수
def plotChart(data): # 외부에서 데이터를 주면 이를 바탕으로 캔들 차트 출력
    chart_style = st.session_state['chart_style']
    marketcolors = mpf.make_marketcolors(up='red',down='blue') # 양, 음봉 선택
    mpf_style = mpf.make_mpf_style(base_mpf_style=chart_style,marketcolors=marketcolors)

    fig, ax = mpf.plot(
        data=data, # 받아온 데이터
        volume=st.session_state['volume'], # 거래량을 출력 여부에 대한 것
        type='candle', # 차트 타입
        style=mpf_style, # 스타일 객체
        figsize=(10,7),
        fontscale=1.1,
        mav=(5,10,30), # 이동평균선(5, 10, 30일 이동평균을 출력하겠다는 뜻)
        mavcolors=('red','green','blue'), # 각 이동평균선의 색상
        returnfig=True # figure 객체 반환 
    )
    st.pyplot(fig)

def has_changes(sliders):
    return any(sliders[key] != initial_values[key] for key in sliders)

# 세션 상태를 초기화 
if 'ndays' not in st.session_state: 
    # 세션 상태에 이미 등록되어 있지 않으면 100일로 초기화 하도록 함
    st.session_state['ndays'] = 100
    
if 'code_index' not in st.session_state:
    # 선택된 종목에 해당하는 정수값을 code_index라는 키로 저장(처음엔 0)
    # 선택된 종목을 세션 상태로 관리
    st.session_state['code_index'] = 0
    
if 'chart_style' not in st.session_state:
    # 차트의 유형은 디폴트로 지정
    st.session_state['chart_style'] = 'default'

if 'volume' not in st.session_state:
    # 거래량 출력 여부는 true 값으로 초기화
    st.session_state['volume'] = True

if 'login_status' not in st.session_state:
    st.session_state['login_status'] = False
    
if 'user_name' not in st.session_state:
    st.session_state['username'] = None

initial_values = {'전자': 0, '에너지': 0, '헬스케어': 0, '차량': 0, '소재': 0, '화학': 0, '미디어': 0, '건설': 0, '금융': 0, '정보기술': 0, '생활소비재': 0, '운송': 0, '중공업': 0, '유통': 0, 'E': 0, 'S': 0, 'G': 0}
if 'sliders' not in st.session_state:
    st.session_state['sliders'] = initial_values.copy()

# if 'form_submitted' not in st.session_state:
#     st.session_state['form_submitted'] = False

with st.sidebar:
    selected = option_menu("메뉴", ['홈','ESG 소개', '방법론','최근 뉴스',
                                  '로그인 / 회원가입','마이페이지'], 
        icons=['bi bi-house','bi bi-globe2','bi bi-map', 'bi bi-newspaper','bi bi-box-arrow-in-right','bi bi-file-earmark-person']
        , menu_icon="cast", default_index=0)


if selected == '홈':
    initial_values = {'전자': 0, '에너지': 0, '헬스케어': 0, '차량': 0, '소재': 0, '화학': 0, '미디어': 0, '건설': 0, '금융': 0, '정보기술': 0, '생활소비재': 0, '운송': 0, '중공업': 0, '유통': 0, 'E': 0, 'S': 0, 'G': 0}
        
    with st.form(key='interest_form'):
        col_1, col_2 = st.columns(2)
        with col_1:
            st.subheader('산업별 관심도')
            for key in initial_values.keys():
                if key not in ['E', 'S', 'G']:
                    st.session_state['sliders'][key] = st.slider(key, 0, 10, st.session_state['sliders'][key], 1)

        with col_2:
            st.subheader('ESG 섹터별 관심도')
            for key in ['E', 'S', 'G']:
                st.session_state['sliders'][key] = st.slider(key, 0, 10, st.session_state['sliders'][key], 1)

        submit_button = st.form_submit_button(label='완료')

    if submit_button:
        if not has_changes(st.session_state['sliders']):
            st.warning('슬라이더 값을 변경하여 주십시오.')
        else:
            st.session_state['form_submitted'] = True
            # 로그인된 사용자의 경우 선호도 저장
            if st.session_state['username']:
                # save_user_preferences(st.session_state['username'], st.session_state['sliders'])
                st.experimental_rerun()
                
            elif st.session_state['username'] is None:
                st.warning('로그인 진행 후 입력하여 주십시오.')
                time.sleep(4)
                st.experimental_rerun()

    # if 'form_submitted' in st.session_state and 'username' in st.session_state:
    if submit_button and 'username' is not None:
        st.success('완료')
        st.write(' ')
        st.subheader('설정 값')
        st.write(' ')
        df = pd.DataFrame(list(st.session_state['sliders'].items()), columns=['항목', '값'])
        st.dataframe(df)

        # 추천 회사 목록 (지금은 예시로 만들어 놓음)
        user_preferences = st.session_state['sliders']
        # recommended_companies = recommend_stocks(user_preferences)
        recommended_companies = ['005930', '000660', '035420']  # 삼성전자, SK하이닉스, NAVER
        df.to_csv('slider_values.csv', index=False)
        st.download_button(label='CSV 다운로드', data=df.to_csv(index=False), file_name='slider_values.csv', mime='text/csv')

        col1, col2 = st.columns([2,3])
        with col1:
            st.subheader('추천 회사 및 정보')
            st.write('''추천 회사
                     ESG 보고서
                     홈페이지''')
        with col2:
            st.subheader('추천 회사의 주가 그래프')
            with st.form(key='chartsetting', clear_on_submit=True):
                st.write('차트 설정')
                symbols = getSymbols()
                recommended_symbols = symbols[symbols['Code'].isin(recommended_companies)]
                choices = list(zip(recommended_symbols.Code, recommended_symbols.Name, recommended_symbols.Market))
                choices = [' : '.join(x) for x in choices]
                if st.session_state['code_index'] >= len(choices):
                    st.session_state['code_index'] = 0
                choice = st.selectbox(label='종목 : ', options=choices, index=st.session_state['code_index'])
                code_index = choices.index(choice)
                code = choice.split()[0]

                ndays = st.slider(label='기간 (days)', min_value=50, max_value=365, value=st.session_state['ndays'], step=1)
                chart_styles = ['default', 'binance', 'blueskies', 'brasil', 'charles', 'checkers', 'classic', 'yahoo', 'mike', 'nightclouds', 'sas', 'starsandstripes']
                chart_style = st.selectbox(label='차트 스타일', options=chart_styles, index=chart_styles.index(st.session_state['chart_style']))
                volume = st.checkbox('거래량', value=st.session_state['volume'])
                st.write('check')
                if st.form_submit_button(label='OK'):
                    st.session_state['ndays'] = ndays
                    st.session_state['code_index'] = code_index
                    st.session_state['chart_style'] = chart_style
                    st.session_state['volume'] = volume

                    data = load_stock_data(code, ndays)
                    plotChart(data)       


        
elif selected == 'ESG 소개':
    col1,_,_ = st.columns([1,2,1])
    with col1:
        st.subheader('**ESG 소개**')
        st.image('https://media.istockphoto.com/id/1447057524/ko/%EC%82%AC%EC%A7%84/%ED%99%98%EA%B2%BD-%EB%B0%8F-%EB%B3%B4%EC%A0%84%EC%9D%84-%EC%9C%84%ED%95%9C-%EA%B2%BD%EC%98%81-esg-%EC%A7%80%EC%86%8D-%EA%B0%80%EB%8A%A5%EC%84%B1-%EC%83%9D%ED%83%9C-%EB%B0%8F-%EC%9E%AC%EC%83%9D-%EC%97%90%EB%84%88%EC%A7%80%EC%97%90-%EB%8C%80%ED%95%9C-%EC%9E%90%EC%97%B0%EC%9D%98-%EA%B0%9C%EB%85%90%EC%9C%BC%EB%A1%9C-%EB%85%B9%EC%83%89-%EC%A7%80%EA%B5%AC%EB%B3%B8%EC%9D%84-%EB%93%A4%EA%B3%A0-%EC%9E%88%EC%8A%B5%EB%8B%88%EB%8B%A4.jpg?s=612x612&w=0&k=20&c=ghQnfLcD5dDfGd2_sQ6sLWctG0xI0ouVaISs-WYQzGA=', width=600)
    st.write("""
        ESG는 환경(Environment), 사회(Social), 지배구조(Governance)의 약자로, 기업이 지속 가능하고 책임 있는 경영을 위해 고려해야 하는 세 가지 핵심 요소를 의미합니다. ESG는 단순한 윤리적 개념을 넘어, 장기적인 기업의 성공과 지속 가능성을 확보하기 위해 중요한 역할을 합니다.

        ### 환경 (Environment)
        환경 요소는 기업이 환경에 미치는 영향을 측정하고 개선하는 데 중점을 둡니다. 이는 기후 변화 대응, 자원 효율성, 오염 방지, 생물 다양성 보전 등의 문제를 포함합니다. 환경 지속 가능성을 강화하는 것은 기업의 평판을 높이고, 법적 리스크를 줄이며, 장기적으로 비용 절감을 가능하게 합니다.

        ### 사회 (Social)
        사회 요소는 기업이 사회에 미치는 영향을 평가합니다. 이는 인권 보호, 노동 조건 개선, 지역 사회 기여, 다양성과 포용성 증진 등을 포함합니다. 긍정적인 사회적 영향을 미치는 기업은 직원의 사기와 생산성을 높이고, 고객과 지역 사회의 신뢰를 얻을 수 있습니다.

        ### 지배구조 (Governance)
        지배구조 요소는 기업의 경영 방식과 의사 결정 과정을 다룹니다. 이는 투명한 회계 관행, 이사회 구성, 경영진의 윤리적 행동, 주주 권리 보호 등을 포함합니다. 건전한 지배구조는 기업의 안정성과 지속 가능성을 보장하고, 투자자들의 신뢰를 증대시킵니다.

        ## 왜 ESG가 중요한가요?
        ### 1. 위험 관리
        ESG를 고려하는 기업은 환경적, 사회적, 법적 리스크를 더 잘 관리할 수 있습니다. 이는 장기적인 기업의 안정성과 성장을 도모합니다.

        ### 2. 투자 유치
        많은 투자자들이 ESG 요인을 고려하여 투자를 결정합니다. ESG를 충실히 이행하는 기업은 더 많은 투자 기회를 얻을 수 있습니다.

        ### 3. 평판 향상
        ESG에 대한 책임을 다하는 기업은 고객과 지역 사회로부터 더 높은 신뢰와 긍정적인 평판을 얻습니다. 이는 브랜드 가치를 높이고, 장기적으로 비즈니스 성공에 기여합니다.

        ### 4. 법적 준수
        전 세계적으로 ESG 관련 규제가 강화되고 있습니다. ESG 기준을 준수하는 기업은 법적 리스크를 최소화하고, 규제 변경에 유연하게 대응할 수 있습니다.

        ## 결론
        ESG는 단순한 트렌드가 아니라, 기업의 지속 가능성과 장기적인 성공을 위한 필수적인 요소입니다. 우리는 ESG 원칙을 바탕으로 책임 있는 경영을 실천하며, 환경 보호, 사회적 기여, 투명한 지배구조를 통해 더 나은 미래를 만들어 나가고자 합니다. 여러분의 지속적인 관심과 지지를 부탁드립니다.
        """)
    
elif selected == '방법론':
    st.write("""
        안녕하십니까 
        당사의 주식 추천 사이트에 방문해 주셔서 감사합니다. 저희는 기업의 환경(Environment), 사회(Social), 지배구조(Governance) 측면을 종합적으로 평가하여 사용자에게 최적의 주식을 추천하는 서비스를 제공합니다. 당사의 방법론은 다음과 같은 주요 요소를 포함합니다.

        ## 1. ESG 스코어 정의 및 평가 기준
        ESG 스코어는 기업의 지속 가능성과 책임 있는 경영을 측정하는 지표로, 다음과 같은 세 가지 주요 분야를 포함합니다:

        #### 환경(Environment)
        기업이 환경 보호를 위해 수행하는 노력과 성과를 평가합니다. 이는 온실가스 배출량, 에너지 효율성, 자원 관리, 재생 가능 에너지 사용 등으로 측정됩니다.

        #### 사회(Social)
        기업의 사회적 책임을 평가합니다. 직원 복지, 지역 사회에 대한 기여, 인권 보호, 공급망 관리 등과 같은 요소가 포함됩니다.

        #### 지배구조(Governance)
        기업의 관리 및 운영 방식에 대한 투명성과 책임성을 평가합니다. 이사회 구조, 경영진의 윤리, 부패 방지 정책, 주주 권리 보호 등이 고려됩니다.

        ## 2. 데이터 수집 및 분석
        저희는 ESG 스코어를 산출하기 위해 신뢰할 수 있는 다양한 데이터 소스를 활용합니다. 주요 데이터 소스에는 기업의 연례 보고서, 지속 가능성 보고서, 뉴스 및 미디어 기사, 그리고 전문 ESG 평가 기관의 리포트가 포함됩니다. 이 데이터를 바탕으로 저희는 다음과 같은 분석 과정을 진행합니다:

        #### 정량적 분석
        수치 데이터 및 KPI(핵심 성과 지표)를 기반으로 한 환경적, 사회적, 지배구조적 성과 분석을 수행합니다.

        #### 정성적 분석
        기업의 정책 및 이니셔티브, 업계 평판 등을 평가하여 ESG 관련 활동의 질적 측면을 분석합니다.

        ## 3. ESG 스코어 산출 및 가중치 적용
        각 기업의 ESG 성과를 기반으로 종합 스코어를 산출하며, 환경, 사회, 지배구조 각 항목에 대해 가중치를 적용하여 전체 ESG 스코어를 계산합니다. 가중치는 산업별, 지역별 특성에 맞추어 조정됩니다. 이 과정에서 기업의 업종과 특성을 반영하여 보다 정확한 평가가 이루어집니다.

        ## 4. 주식 추천 알고리즘
        ESG 스코어를 바탕으로 사용자 맞춤형 주식 추천 알고리즘을 운영합니다. 사용자의 투자 목표, 리스크 수용 범위, 관심 산업 등을 고려하여 ESG 점수가 높은 기업을 추천합니다. 알고리즘은 다음과 같은 요소를 반영합니다:

        #### ESG 스코어
        높은 ESG 스코어를 가진 기업을 우선 추천합니다.
        #### 재무 성과
        기업의 재무 건전성과 성장 잠재력도 함께 고려합니다.
        #### 시장 동향
        현재 시장 동향 및 산업별 특성을 반영하여 추천합니다.
    
        ## 5. 지속적인 모니터링 및 업데이트
        ESG 관련 정보는 지속적으로 업데이트되며, 기업의 ESG 스코어는 정기적으로 재평가됩니다. 이를 통해 최신 정보를 바탕으로 사용자에게 정확한 추천을 제공하며, 기업의 ESG 성과 변화에 신속하게 대응합니다.

        ## 6. 투명한 정보 제공
        저희는 사용자가 신뢰할 수 있는 정보를 제공하기 위해 ESG 스코어 산출 과정과 데이터 출처를 투명하게 공개합니다. 사용자는 각 기업의 ESG 성과에 대한 자세한 정보를 확인할 수 있으며, 이를 바탕으로 보다 나은 투자 결정을 내릴 수 있습니다.

        저희의 ESG 스코어 기반 주식 추천 서비스는 책임 있는 투자와 지속 가능한 성장을 지향합니다. 여러분의 투자 결정에 도움이 되기를 바랍니다.""")
    
elif selected == '최근 뉴스':
    st.write(' ')
    st.write(' ')
    st.subheader('최근 경제 뉴스')

elif selected == '로그인 / 회원가입':
    with open('config.yaml','r',encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    
    try:
        authenticator.login()
    except LoginError as e:
        st.error(e)    
    
    if st.session_state["authentication_status"]:
        authenticator.logout()
        st.write(f'로그인을 성공하셨습니다. *{st.session_state["name"]}* 님')
        st.write('마이페이지를 볼 수 있습니다.')
        st.session_state['username'] = st.session_state['name']
        # st.experimental_rerun()
    elif st.session_state["authentication_status"] is False:
        st.error('등록된 ID가 없습니다.')
        # submit = st.form_submit_button('회원가입')  
        register_button = st.button(label = '회원가입')
        # if register_button:
    elif st.session_state["authentication_status"] is None:
        st.warning('ID와 PW를 입력하여 주십시오.')

    # pw 리셋 버튼
    if st.session_state["authentication_status"]:
        try:
            if authenticator.reset_password(st.session_state["username"]):
                st.success('Password modified successfully')
        except ResetError as e:
            st.error(e)
        except CredentialsError as e:
            st.error(e)

    # 회원가입에 대한 위젯
    try:
        (email_of_registered_user,
        username_of_registered_user,
        name_of_registered_user) = authenticator.register_user(pre_authorization=False)
        if email_of_registered_user:
            st.success('회원가입이 완료되었습니다.')
    except RegisterError as e:
        st.error(e)

# Creating a forgot password widget
    try:
        (username_of_forgotten_password,
        email_of_forgotten_password,
        new_random_password) = authenticator.forgot_password()
        if username_of_forgotten_password:
            st.success('New password sent securely')
        # Random password to be transferred to the user securely
        elif not username_of_forgotten_password:
            st.error('Username not found')
    except ForgotError as e:
        st.error(e)

    # id 까먹었을 때
    try:
        (username_of_forgotten_username,
        email_of_forgotten_username) = authenticator.forgot_username()
        if username_of_forgotten_username:
            st.success('ID가 등록되었습니다.')
            # Username to be transferred to the user securely
        elif not username_of_forgotten_username:
            st.error('등록된 email을 찾을 수 없습니다.')
    except ForgotError as e:
        st.error(e)

    # Creating an update user details widget
    if st.session_state["authentication_status"]:
        try:
            if authenticator.update_user_details(st.session_state["username"]):
                st.success('Entries updated successfully')
        except UpdateError as e:
            st.error(e)

    # 해당 회원가입에 대한 정보 저장
    with open('config.yaml', 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False)

        
elif selected == '마이페이지':
    if st.session_state['login_status'] == False:
        st.warning('로그인을 진행하여 주십시오.')
    if st.session_state['login_status'] == True:
        st.write('마이 페이지')