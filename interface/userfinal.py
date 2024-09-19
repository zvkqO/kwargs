import streamlit as st
from bs4 import BeautifulSoup
import requests
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service  # Service 클래스 임포트
import pandas as pd
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
from streamlit_authenticator.utilities.hasher import Hasher
import os.path
import pickle as pkle
from streamlit_js_eval import streamlit_js_eval
from passlib.context import CryptContext
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events
from cvxopt import matrix, solvers
from streamlit_authenticator.utilities import (CredentialsError,
                                               ForgotError,
                                               Hasher,
                                               LoginError,
                                               RegisterError,
                                               ResetError,
                                               UpdateError)
from pymongo import MongoClient
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(
        page_title="ESG 정보 제공 플랫폼",
        page_icon=":earth_africa:",
        layout="wide",
    )
st.title('Kwargs')

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

if 'clicked_points' not in st.session_state:
    st.session_state['clicked_points'] = None

if 'sliders' not in st.session_state:
    st.session_state['sliders'] = {}

for key in ['environmental', 'social', 'governance']:
    if key not in st.session_state['sliders']:
        st.session_state['sliders'][key] = 0

# MongoDB 연결 설정
client = MongoClient("mongodb+srv://tlsgofl0404:Xfce0WwgjDGFx7YH@kwargs.9n9kn.mongodb.net/?retryWrites=true&w=majority&appName=kwargs")
db = client['kwargs']
collection = db['kwargs']

# 전처리 함수 정의
def preprocess_data(df):
    # 기존 컬럼명을 사용할 수 있도록 유효성을 확인
    if 'environmental' in df.columns and 'social' in df.columns and 'governance' in df.columns:
        # ESG 영역 비중을 백분율로 환산
        df['env_percent'] = df['environmental'] / (df['environmental'] + df['social'] + df['governance'])
        df['soc_percent'] = df['social'] / (df['environmental'] + df['social'] + df['governance'])
        df['gov_percent'] = df['governance'] / (df['environmental'] + df['social'] + df['governance'])

        # 각 영역별 최종 점수 계산 (average_label 필요)
        df['env_score'] = df['average_label'] * df['env_percent']
        df['soc_score'] = df['average_label'] * df['soc_percent']
        df['gov_score'] = df['average_label'] * df['gov_percent']

        # 연도별 가중치 설정
        latest_year = df['Year'].max()
        year_weights = {
            latest_year: 0.5,
            latest_year - 1: 0.25,
            latest_year - 2: 0.125,
            latest_year - 3: 0.0625,
            latest_year - 4: 0.0625
        }

        # 가중치를 반영한 각 영역별 점수 합산
        df['environmental'] = df.apply(lambda x: x['env_score'] * year_weights.get(x['Year'], 0), axis=1)
        df['social'] = df.apply(lambda x: x['soc_score'] * year_weights.get(x['Year'], 0), axis=1)
        df['governance'] = df.apply(lambda x: x['gov_score'] * year_weights.get(x['Year'], 0), axis=1)

        # 동일 기업의 연도별 점수를 합산하여 최종 점수 도출
        final_df = df.groupby(['Company', 'industry', 'ticker']).agg({
            'environmental': 'sum',
            'social': 'sum',
            'governance': 'sum'
        }).reset_index()

        return final_df
    else:
        raise KeyError("The expected columns 'environmental', 'social', and 'governance' are not present in the dataframe.")


# step 1 : load the provided dataset
file_path = r"C:\esgpage\kwargs\esgpage\userinterface\240820_final_dummy.csv"
dummy = pd.read_csv(file_path, encoding='euc-kr')
dummy = preprocess_data(dummy)

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

# 캔들차트 출력 함수
def plotChart(data): # 외부에서 데이터를 주면 이를 바탕으로 캔들 차트 출력
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
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

# 크롤링  필요한 함수 정의
def setup_webdriver():
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument('--headless')  # UI 없이 실행하기 위한 headless 모드
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    # 서비스 객체를 사용하여 드라이버 초기화
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(3)
    
    return driver

# 필요한 함수 정의
def setup_webdriver():
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument('--headless')  # UI 없이 실행하기 위한 headless 모드
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    # 서비스 객체를 사용하여 드라이버 초기화
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(3)
    
    return driver

# URL 생성 함수 정의
def makePgNum(num):
    return 1 + 10 * (num - 1)

def makeUrl(search, page):
    page_num = makePgNum(page)
    url = f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query={search}&start={page_num}"
    return url

# 뉴스 크롤링 함수 추가
def crawl_naver_news(search):
    driver = setup_webdriver()

    target_article_count = 10
    collected_article_count = 0
    current_page = 1
    naver_urls = []

    while collected_article_count < target_article_count:
        search_url = makeUrl(search, current_page)
        driver.get(search_url)
        time.sleep(1)  # 대기시간 변경 가능

        a_tags = driver.find_elements(By.CSS_SELECTOR, 'a.info')

        for a_tag in a_tags:
            if collected_article_count >= target_article_count:
                break
            a_tag.click()
            driver.switch_to.window(driver.window_handles[1])
            time.sleep(3)

            url = driver.current_url
            if "news.naver.com" in url:
                naver_urls.append(url)
                collected_article_count += 1

            driver.close()
            driver.switch_to.window(driver.window_handles[0])

        current_page += 1

    driver.quit()
    return fetch_news_contents(naver_urls)

def fetch_news_contents(naver_urls):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}
    news_list = []

    for url in naver_urls:
        original_html = requests.get(url, headers=headers)
        html = BeautifulSoup(original_html.text, "html.parser")
        title_element = html.select_one("div#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
        title = title_element.get_text(strip=True) if title_element else "No title found"
        news_list.append((title, url))

    return news_list




# step2-1 : 사용자 입력
def get_user_input():
    with st.form(key='interest_form'):
        # 사용자의 ESG 선호도와 관심 산업군을 입력받는 부분입니다.
        industry_choices = df_new['industry'].unique().tolist()
        selected_industries = st.multiselect('관심 산업군을 선택하세요',industry_choices,key='unique_key_for_industries')
        esg_weights = {}
        for key in ['environmental', 'social', 'governance']:
            st.session_state['sliders'][key] = st.slider(key, 0, 10, st.session_state['sliders'][key], 1)
            esg_weights[key] = st.session_state['sliders'][key]
        # with col_2:
        #     esg_weights = {}
        #     for key in ['environmental', 'social', 'governance']:
        #         st.session_state['sliders'][key] = st.slider(key, 0, 10, st.session_state['sliders'][key], 1)
        #         esg_weights[key] = st.session_state['sliders'][key]
        submit_button = st.form_submit_button(label='완료')
        
    if submit_button:
        all_sliders_zero = all(value == 0 for value in st.session_state['sliders'].values())
        
        # 조건 검사: multiselect에서 아무것도 선택하지 않았거나 슬라이더 값이 모두 0인 경우
        if not selected_industries or all_sliders_zero:
            st.warning('슬라이더 값을 변경하여 주십시오.')
        else:
            st.write(' ')
            st.write(' ')
            return esg_weights, selected_industries
    return esg_weights, selected_industries

# Step 3: 기업 추천
def recommend_companies(esg_weights, selected_industries, df):
    # 전처리된 데이터에서 사용자의 ESG 선호도 가중치를 반영하여 최종 점수 계산
    df['final_score'] = (
        esg_weights['environmental'] * df['environmental'] +
        esg_weights['social'] * df['social'] +
        esg_weights['governance'] * df['governance']
    )

    # 산업군 필터링
    filtered_df = df[df['industry'].isin(selected_industries)]

    # 상위 10개 기업 선정
    top_companies = filtered_df.sort_values(by='final_score', ascending=False).head(10)

    return top_companies


# Step 4: 포트폴리오 비중 계산
def calculate_portfolio_weights(top_companies):
    # 추천된 상위 10개 기업을 바탕으로 포트폴리오 비중을 계산합니다.
    # 입력:
    # - top_companies: 상위 10개 기업 데이터프레임 (ticker 포함)

    tickers = top_companies['ticker'].tolist()
    price_data = yf.download(tickers, start="2019-01-01", end="2023-01-01")['Adj Close']

    if price_data.isnull().values.any():
        return "일부 데이터가 누락되었습니다. 다른 기업을 선택해 주세요.", None

    # 일별 수익률 계산
    returns = price_data.pct_change().dropna()

    # 누적 기대 수익률 계산(추가작성)
    cumulative_returns = (1 + returns).cumprod() - 1
    
    # 평균 수익률과 공분산 행렬
    mu = returns.mean().values
    Sigma = returns.cov().values
    # mu = expected_returns.mean_historical_return(price_data)
    # S = risk_models.sample_cov(price_data)

    # ef = EfficientFrontier(mu, S)
    # weights = ef.max_sharpe()
    # cleaned_weights = ef.clean_weights()
    # performance = ef.portfolio_performance(verbose=True)

    # # 출력: 포트폴리오 비중 (dict) 및 성과 (tuple)
    # return cleaned_weights, performance
        # `cvxopt`에서 사용할 행렬 형식으로 변환
    n = len(mu)
    P = matrix(Sigma)
    q = matrix(np.zeros(n))
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    A = matrix(1.0, (1, n))
    b = matrix(1.0)

    # 쿼드라틱 프로그래밍 솔버 실행
    sol = solvers.qp(P, q, G, h, A, b)

    # 최적 가중치 추출
    weights = np.array(sol['x']).flatten()

    # 포트폴리오 성과 지표 계산
    expected_return = np.dot(weights, mu)
    expected_volatility = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
    sharpe_ratio = expected_return / expected_volatility

    # 가중치 정리
    cleaned_weights = dict(zip(tickers, weights))
    


    return cleaned_weights,(expected_return, expected_volatility, sharpe_ratio), cumulative_returns

def calculate_expected_returns(ticker, start_date="2019-01-01", end_date="2023-01-01"):
    # 주가 데이터 로드
    data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    
    # 일별 수익률 계산
    returns = data.pct_change().dropna()
    
    # 누적 기대 수익률 계산
    cumulative_returns = (1 + returns).cumprod() - 1
    
    return cumulative_returns

with st.sidebar:
    # selected = option_menu("메뉴", ['홈','ESG 소개', '방법론','최근 뉴스',
    #                               '로그인 / 회원가입','마이페이지'], 
    #     icons=['bi bi-house','bi bi-globe2','bi bi-map', 'bi bi-newspaper','bi bi-box-arrow-in-right','bi bi-file-earmark-person']
    #     , menu_icon="cast", default_index=0)
    selected = option_menu("메뉴", ['홈','ESG 소개', '방법론','최근 뉴스'], 
        icons=['bi bi-house','bi bi-globe2','bi bi-map', 'bi bi-newspaper']
        , menu_icon="cast", default_index=0)

industry_trends = {
    'Technology': """
    <ul>
        <li><strong>인공지능과 머신러닝의 진화 </strong>
            <p>인공지능(AI)과 머신러닝(ML) 기술은 더욱 고도화되며, 다양한 산업에 적용되고 있습니다. AI는 데이터 분석, 자동화, 예측 분석에서 중요한 역할을 하며, 특히 자연어 처리(NLP), 이미지 인식, 자율주행차 등에서 그 응용 범위가 확장되고 있습니다.</p>
        </li>
        <li><strong>클라우드 컴퓨팅의 확산 </strong>
            <p>기업들은 IT 자원의 유연성과 효율성을 위해 클라우드 기반 솔루션을 채택하고 있습니다. 하이브리드 클라우드와 멀티클라우드 전략이 인기를 끌며, 데이터 보안과 컴플라이언스 문제에 대한 해결책이 계속해서 발전하고 있습니다.</p>
        </li>
        <li><strong>엣지 컴퓨팅의 부상 </strong>
            <p>IoT(사물인터넷) 장치의 증가와 함께 엣지 컴퓨팅의 중요성이 커지고 있습니다. 데이터 처리를 클라우드가 아닌 데이터 생성 지점 근처에서 수행함으로써, 지연(latency)을 줄이고 실시간 분석을 가능하게 합니다.</p>
        </li>
    </ul>
    """,
    'Automobile': """
    <ul>
        <li><strong>전기차(EV)의 확산 </strong>
            <p>전기차는 환경 규제 강화와 소비자의 친환경적 요구에 맞춰 빠르게 성장하고 있습니다. 배터리 기술의 발전과 충전 인프라의 확충이 전기차 시장의 주요 동력으로 작용하고 있습니다.</p>
        </li>
        <li><strong>자율주행 기술의 발전 </strong>
            <p>자율주행차 기술은 레이더, LiDAR, 카메라와 같은 센서를 통합하여 도로 안전성을 높이고 있습니다. 다양한 기업들이 자율주행 기술의 상용화에 박차를 가하고 있으며, 법적, 윤리적 문제 해결이 중요한 과제로 떠오르고 있습니다.</p>
        </li>
        <li><strong>커넥티드 카의 진화 </strong>
            <p>차량 간 통신(V2V) 및 차량과 인프라 간 통신(V2I)이 향후 자동차 산업의 주요 트렌드로 자리잡고 있습니다. 이는 교통 혼잡도 감소, 사고 예방, 연비 개선 등 다양한 이점을 제공합니다.</p>
        </li>
    </ul>
    """,
    'Energy': """
    <ul>
        <li><strong>재생 에너지의 확대  </strong>
            <p>태양광, 풍력 등 재생 가능한 에너지원이 급격히 확산되고 있습니다. 이는 정부의 탄소 중립 정책과 기술 혁신에 힘입어 에너지 믹스의 중요한 부분으로 자리 잡고 있습니다.</p>
        </li>
        <li><strong>에너지 저장 기술의 발전  </strong>
            <p>에너지 저장 시스템(ESS)의 발전은 재생 에너지의 간헐성 문제를 해결하는 데 기여하고 있습니다. 리튬 이온 배터리와 같은 고성능 저장 기술의 발전이 지속적으로 이루어지고 있습니다.</p>
        </li>
        <li><strong>스마트 그리드 기술 </strong>
            <p>전력망의 효율성과 안정성을 높이기 위한 스마트 그리드 기술이 주목받고 있습니다. 이는 전력 소비 패턴을 실시간으로 모니터링하고, 수요와 공급을 효과적으로 조절하는 데 도움을 줍니다.</p>
        </li>
    </ul>
    """,
    'Finance': """
    <ul>
        <li><strong>핀테크의 혁신  </strong>
            <p>디지털 결제, 블록체인 기술, 암호화폐 등이 금융 산업의 혁신을 주도하고 있습니다. 특히 블록체인은 거래의 투명성을 높이고, 핀테크는 금융 서비스의 접근성을 개선하고 있습니다.</p>
        </li>
        <li><strong>AI 기반 리스크 관리 </strong>
            <p>인공지능을 활용한 리스크 분석 및 관리가 금융 산업에서 중요해지고 있습니다. AI는 대량의 데이터를 분석하여 잠재적 리스크를 사전에 식별하고, 보다 정확한 의사결정을 지원합니다.</p>
        </li>
        <li><strong>디지털 자산과 암호화폐의 성장  </strong>
            <p>암호화폐와 디지털 자산의 시장이 확대됨에 따라, 규제와 보안 문제가 주요 이슈로 대두되고 있습니다. 또한, 중앙은행 디지털 화폐(CBDC)의 도입이 금융 시스템에 미치는 영향이 주목받고 있습니다.</p>
        </li>
    </ul>
    """,
    'Construction': """
    <ul>
        <li><strong>스마트 건설 기술의 도입 </strong>
            <p>BIM(건물 정보 모델링), IoT 센서, 드론 등이 스마트 건설의 핵심 기술로 자리잡고 있습니다. 이는 건설 과정의 효율성을 높이고, 실시간 모니터링 및 데이터 기반 의사결정을 가능하게 합니다.</p>
        </li>
        <li><strong>지속 가능한 건축 및 그린 빌딩 </strong>
            <p>환경 친화적 건축 자재와 에너지 효율성이 높은 설계가 지속 가능한 건축의 주요 트렌드로 부각되고 있습니다. 이는 환경 규제 강화와 소비자들의 친환경적 요구에 대응하기 위한 전략입니다.</p>
        </li>
        <li><strong>3D 프린팅과 모듈러 건설 </strong>
            <p>3D 프린팅 기술을 활용한 건축 자재 제작과 모듈러 건설 방식이 건설 산업의 혁신을 이끌고 있습니다. 이는 비용 절감과 공사 기간 단축에 기여하고 있습니다.</p>
        </li>
    </ul>
    """,
    'Materials': """
    <ul>
        <li><strong>고성능 신소재 개발 </strong>
            <p>탄소 나노튜브, 그래핀 등 고성능 신소재의 개발이 각광받고 있습니다. 이러한 신소재는 전자기기, 에너지 저장 장치, 자동차 부품 등 다양한 분야에서 응용되고 있습니다.</p>
        </li>
        <li><strong>재활용 및 지속 가능성 </strong>
            <p>자원 고갈 문제와 환경 문제 해결을 위한 재활용 및 지속 가능한 소재의 개발이 중요한 트렌드로 떠오르고 있습니다. 특히 플라스틱 재활용 기술과 바이오 기반 소재가 주목받고 있습니다.</p>
        </li>
        <li><strong>스마트 소재의 상용화 </strong>
            <p>자가 치유 기능, 환경에 반응하는 스마트 소재 등이 연구되고 있으며, 이는 다양한 산업 분야에서 혁신적인 응용 가능성을 열어주고 있습니다.</p>
        </li>
    </ul>
    """,
    'Healthcare': """
    <ul>
        <li><strong>디지털 헬스케어의 발전 </strong>
            <p>원격 진료, 모바일 헬스 애플리케이션, 웨어러블 기기 등이 헬스케어 분야에서 중요한 역할을 하고 있습니다. 이들 기술은 환자의 건강 모니터링과 치료 접근성을 향상시키는 데 기여하고 있습니다.</p>
        </li>
        <li><strong>정밀 의학과 유전자 분석 </strong>
            <p>정밀 의학은 개인의 유전자 정보를 기반으로 맞춤형 치료를 제공하고 있습니다. 유전자 분석 기술의 발전은 질병의 조기 발견과 효과적인 치료를 가능하게 하고 있습니다.</p>
        </li>
        <li><strong>AI와 머신러닝을 활용한 진단 및 치료 </strong>
            <p>AI와 머신러닝 기술이 의료 영상 분석, 질병 예측, 치료 계획 수립 등에서 사용되고 있습니다. 이는 진단의 정확도를 높이고, 효율적인 치료 방법을 제시하는 데 도움을 줍니다.</p>
        </li>
    </ul>
    """
}
if selected == '홈':
    df_new = dummy.copy()
    initial_values = {'E': 0, 'S': 0, 'G': 0}
    col_1, col_2, alpha = st.columns([3,3,1])
    with col_1:
        esg_weight, selected_industries = get_user_input() # esg 선호도, 관심 산업군

    if esg_weight and selected_industries:
    # 주식 추천 알고리즘 -> 추천 회사 도출
        top_companies = recommend_companies(esg_weight,selected_industries,df_new)
    
    # 포트폴리오 비중 계산
    # cleaned_weights:각 자산에 할당된 최적의 투자 비율
    # performance:최적화된 포트폴리오의 성과 지표
        cleaned_weights, performance, cumulative_returns = calculate_portfolio_weights(top_companies)
        top_companies['Weight'] = top_companies['ticker'].map(cleaned_weights)
        # top_companies['Expected Return'] = top_companies['ticker'].map(expected_returns_dict)
        
        with col_2:
            # st.subheader('추천 회사')
            top_companies = top_companies.sort_values(by='Weight', ascending=False)
            st.markdown(f"""
                <div>
                    <h3 style="color:#333; text-align:center; font-size:24px">추천 회사</h3>
                </div>
                """, unsafe_allow_html=True)
            fig = px.pie(top_companies, names='Company', values='Weight', color_discrete_sequence=px.colors.qualitative.G10)
            fig.update_traces(textposition='auto',customdata=top_companies['Weight'] / top_companies['Weight'].sum() * 100,textinfo='percent+label+value',hovertemplate='%{label}: %{customdata:.1f}%',texttemplate='%{label}',)
            fig.update_layout(font=dict(size=16, color='black'),showlegend=False,
                legend=dict(orientation="v",yanchor="middle",y=0.5,xanchor="left",x=1.1  # Position it outside the pie chart area on the right side
        ),margin=dict(t=40, b=40, l=0, r=0),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)', width=600,height=400,)

            clicked_points = plotly_events(fig, click_event=True,key="company_click")
        with alpha:
            st.write(' ')
            st.write(' ')
            companies = top_companies['Company'].unique()
            output = '<div>'
            order = 1
            other_percent = 0
            top_companies = top_companies.sort_values(by='Weight', ascending=False)
            for i, row in top_companies.iterrows():
                weight_percent = row['Weight'] * 100
                if weight_percent >= 5:
                    weight_percent = round(weight_percent, 2)
                    st.markdown( f'''
                    <ul style="font-size:17px; letter-spacing: 1.3px;">
                        <span style="margin-right: 10px;">{order}. {row["Company"]}</span>
                        <span>{weight_percent}%</span>
                    </ul>
                    ''', unsafe_allow_html=True) 
                    order += 1
                else:
                    other_percent += weight_percent
            order += 1
            other_percent = round(other_percent, 2)
            st.markdown(f'''
                    <ul style="font-size:17px; letter-spacing: 1.3px;">
                        <span style="margin-right: 10px;">{order}. 기타</span>
                        <span>{other_percent}%</span>
                    </ul>
                    ''', unsafe_allow_html=True)
            
        col_3, col_4,col_5 = st.columns([3,3,4])
        with col_3:
            if clicked_points:
                clicked_point = clicked_points[0]
                if 'pointNumber' in clicked_point:
                    company_index = clicked_point['pointNumber']
                    if company_index < len(top_companies):
                        company_info = top_companies.iloc[company_index]
                        clicked_company = company_info['Company']
                        # 📊 <div style="background-color:#f9f9f9; padding:20px; border-radius:10px; border: 1px solid #e0e0e0;">
                        # color:#666; 
                        st.markdown(f"""
                <div>
                    <h3 style="color:#333; text-align:center; font-size:24px">{clicked_company} ({company_info['industry']})</h3>
                    <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                        <div style="flex:1; text-align:center; padding:10px;">
                            <h4 style="font-weight:bold; font-size:20px;">환경</h4>
                                <p style="font-size:20px; color:#444;font-weight:bold;text-align:center;">{company_info['environmental']:.2f}</p>
                        </div>
                        <div style="flex:1; text-align:center; padding:10px;">
                            <h4 style="font-weight:bold;font-size:20px;">사회</h4>
                                <p style="font-size:20px; color:#444;font-weight:bold;text-align:center;">{company_info['social']:.2f}</p>
                        </div>
                        <div style="flex:1; text-align:center; padding:10px;">
                            <h4 style="font-weight:bold;font-size:20px;">지배구조</h4>
                                <p style="font-size:20px; color:#444;font-weight:bold;text-align:center;">{company_info['governance']:.2f}</p>
                        </div>
                    </div>
                    <div style="text-align:center; margin-top:10px;">
                        <h4 style="font-size:22px; font-weight:bold;">추천 포트폴리오 비중</h4>
                        <p style="font-size:22px; font-weight:bold;">{company_info['Weight'] * 100:.2f}%</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.write(' ')
                
    
            st.write(' ')
            st.write(' ')
            st.write(' ')
        
        with col_4:
            try:
                # st.subheader(f'{clicked_company} 주가 그래프')
                st.markdown(f"""<div>
                            <h2 style="font-size: 24px; text-align:center;">{clicked_company} 주가 그래프</h2>
                            </div>
                """, unsafe_allow_html=True)
                
                
                with st.form(key='chartsetting', clear_on_submit=True):
                # top_companies에서 Company와 Ticker 열 사용
                    company_choices = top_companies['Company'].tolist()
                    ticker_choices = top_companies['ticker'].tolist()
                    ticker_choices = [ticker.replace('.KS', '') for ticker in ticker_choices]

                    if st.session_state['code_index'] >= len(company_choices):
                        st.session_state['code_index'] = 0

            # 회사명으로 선택, 하지만 실제로는 Ticker 값을 사용
                # choice = st.selectbox(label='종목 : ', options=company_choices, index=st.session_state['code_index'])
                    choice = clicked_company
                    code_index = company_choices.index(choice)
                    code = ticker_choices[code_index]  # 선택한 회사의 Ticker 값을 가져옴

            # 슬라이더로 기간 선택
                    ndays = st.slider(label='기간 (days)', min_value=50, max_value=365, value=st.session_state['ndays'], step=1)
                    chart_styles = ['default', 'binance', 'blueskies', 'brasil', 'charles', 'checkers', 'classic', 'yahoo', 'mike', 'nightclouds', 'sas', 'starsandstripes']
                    chart_style = st.selectbox(label='차트 스타일', options=chart_styles, index=chart_styles.index(st.session_state['chart_style']))
                    volume = st.checkbox('거래량', value=st.session_state['volume'])
        
            # 폼 제출 버튼
                    if st.form_submit_button(label='OK'):
                # 세션 상태 업데이트
                        st.session_state['ndays'] = ndays
                        st.session_state['code_index'] = code_index
                        st.session_state['chart_style'] = chart_style
                        st.session_state['volume'] = volume

                # 선택된 종목의 주가 데이터 로드
                        data = load_stock_data(code, ndays)
                
                # 주가 차트 시각화 함수 호출
                        plotChart(data)
            except:
                st.write(' ')
                
        with col_5:
            if clicked_points:
                st.markdown(f"""<div>
                            <h2 style="font-size: 24px; text-align:center;">{clicked_company} 워드 클라우드</h2>
                            </div>
                """, unsafe_allow_html=True)
                
                # MongoDB에서 Company_ID 입력을 받아 해당 데이터를 불러오기
                Company = clicked_company

                # MongoDB에서 해당 Company_ID의 제목들 불러오기
                titles = collection.find({'Company': Company}, {'_id': 0, 'title': 1})
                st.write(titles)
                
                # 불러온 데이터 리스트로 저장
                title_list = [document['title'] for document in titles if 'title' in document]
                st.write(title_list)
                # 형태소 분석기 설정
                okt = Okt()
                nouns_adj_verbs = []

                for title in title_list:
                    # 명사(N), 형용사(Adjective)만 추출
                    tokens = okt.pos(title, stem=True)
                    for word, pos in tokens:
                        if pos in ['Noun', 'Adjective']:
                            nouns_adj_verbs.append(word)

                # 빈도수 계산
                word_counts = Counter(nouns_adj_verbs)
                st.write(word_counts)
                data = word_counts.most_common(500)
                tmp_data = dict(data)
                st.write(tmp_data)
                
                # 워드 클라우드 생성
                wordcloud = WordCloud(
                    font_path='/usr/share/fonts/truetype/nanum/NanumPen.ttf',  # 한글 폰트 설정
                    background_color='white',
                    width=800,
                    height=600
                ).generate_from_frequencies(tmp_data)

                # 워드 클라우드 시각화 및 출력
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')

                # Streamlit에 워드 클라우드 출력
                st.pyplot(fig)

   

        
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

    # 검색어 입력
    search = st.text_input("검색할 키워드를 입력하세요:")

    # 버튼 클릭 시 크롤링 시작
    if st.button("뉴스 검색"):
        if search:
            st.write(f"'{search}' 관련 기사를 검색 중입니다...")
            news_list = crawl_naver_news(search)

            if news_list:
                # st.write(f"수집된 기사 수: {len(news_list)}개")
                for title, link in news_list:
                    st.markdown(f"- [{title}]({link})")
            else:
                st.write("기사를 찾을 수 없습니다.")
        else:
            st.write("검색어를 입력해주세요.")