import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import FinanceDataReader as fdr
import mplfinance as mpf
from datetime import datetime, timedelta
import json
import requests as rq
import bs4
import matplotlib.pyplot as plt
import pandas as pd

# _,col,_ = st.columns([1,1,1])
# with col:
#     st.title('Kwargs')
st.set_page_config(
        page_title="ESG 정보 제공 플랫폼",
        page_icon=":earth_africa:",
        layout="centered",
    )
st.title('Kwargs')

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
    # 출력되는 기간에 해당하는 ndays 키 관리
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

initial_values = {'전자': 0, '에너지': 0, '헬스케어': 0, '차량': 0, '소재': 0, '화학': 0, '미디어': 0, '건설': 0, '금융': 0, '정보기술': 0, '생활소비재': 0, '운송': 0, '중공업': 0, '유통': 0, 'E': 0, 'S': 0, 'G': 0}

if 'sliders' not in st.session_state:
    st.session_state['sliders'] = initial_values.copy()

def news_crawling():
    url = f''
    r = requests.get(url)
    soup = BeautifulSoup(r.text)
    

with st.sidebar:
    selected = option_menu("메뉴", ['홈','ESG 소개', '방법론','최근 뉴스',
                                  '로그인 / 회원가입','마이페이지'], 
        icons=['bi bi-house','bi bi-globe2','bi bi-map', 'bi bi-newspaper','bi bi-box-arrow-in-right','bi bi-file-earmark-person']
        , menu_icon="cast", default_index=0)


if selected == '홈':
    # initial_values = {'전자': 0,'에너지': 0,'헬스케어': 0,'차량': 0,
    # '소재': 0,'화학': 0,'미디어': 0,'건설': 0,'금융': 0,'정보기술': 0,
    # '생활소비재': 0,'운송': 0,'중공업': 0,'유통': 0,
    # 'E': 0,'S': 0,'G': 0 }

    with st.form(key='interest_form'):
        st.subheader('산업별 관심도')
        for key in initial_values.keys():
            if key not in ['E', 'S', 'G']:
                st.session_state['sliders'][key] = st.slider(key, 0, 10, st.session_state['sliders'][key], 1)
        
        st.markdown('---')
        st.subheader('ESG 섹터별 관심도')
        for key in ['E', 'S', 'G']:
            st.session_state['sliders'][key] = st.slider(key, 0, 10, st.session_state['sliders'][key], 1)

        submit_button = st.form_submit_button(label='완료')
    if submit_button:
        if not has_changes(st.session_state['sliders']):
            st.warning('슬라이더를 변경하여 주십시오.')
        else:
            st.session_state['form_submitted'] = True
            st.experimental_rerun()

    if 'form_submitted' in st.session_state:
        st.success('완료')
        st.write('관심도')
        st.write(' ')
        df = pd.DataFrame(list(st.session_state['sliders'].items()), columns=['항목', '값'])
        st.dataframe(df)
        # 추천 회사 목록 예시
        recommended_companies = ['005930', '000660', '035420']  # 삼성전자, SK하이닉스, NAVER
            
        df.to_csv('slider_values.csv', index=False)
        st.download_button(label='CSV 다운로드', data=df.to_csv(index=False), file_name='slider_values.csv', mime='text/csv')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('추천 회사 및 정보')
            st.write('추천 회사, ESG 보고서, 홈페이지 제공')
            
        with col2:
            st.subheader('추천 회사의 주가 그래프')
            with st.form(key='chartsetting', clear_on_submit=True):
                st.write('차트 설정')
                symbols = getSymbols()
                recommended_symbols = symbols[symbols['Code'].isin(recommended_companies)]
                choices = list(zip(recommended_symbols.Code, recommended_symbols.Name, recommended_symbols.Market))
                choices = [' : '.join(x) for x in choices]
                if st.session_state['code_index'] >= len(choices):
                    st.session_state['code_index'] = 0  # 선택 가능한 범위 내로 인덱스 조정
                choice = st.selectbox(label='종목 : ', options=choices, index=st.session_state['code_index'])
                code_index = choices.index(choice)
                code = choice.split()[0]

                ndays = st.slider(label='기간 (days)', min_value=50, max_value=365, value=st.session_state['ndays'], step=1)
                chart_styles = ['default', 'binance', 'blueskies', 'brasil', 'charles', 'checkers', 'classic', 'yahoo', 'mike', 'nightclouds', 'sas', 'starsandstripes']
                chart_style = st.selectbox(label='차트 스타일', options=chart_styles, index=chart_styles.index(st.session_state['chart_style']))
                volume = st.checkbox('거래량', value=st.session_state['volume'])

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
    st.write('방법방법')
# elif selected == '최근 뉴스':
# elif selected == '로그인 / 회원가입':
# elif selected == '마이페이지':


