from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (ChatPromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate,
                               HumanMessagePromptTemplate,)
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.chains import ConversationChain

import os
import pandas as pd
import datetime as dt
import streamlit as st
from streamlit_chat import message

k=4
conversation_key = "conversation"
human_message_key = "human"
df = pd.read_csv('policy_data231130.csv')

###### 대화용 AI ######
def getConversation():
    # api key 세팅
    os.environ["OPENAI_API_KEY"] = st.secrets["API_KEY"]

    # 시스템 설정: 역할부여 정의
    system_message = SystemMessagePromptTemplate.from_template("""
    너는 사용자에게 정책을 찾아주는 자동응답기야.
    사용자가 너에게 인사를 하면 너는 '안녕하세요! 저는 청플리입니다. 당신에게 맞는 정책을 찾아드릴게요.'라고 하고,
    사용자의 <나이>를 물어봐야 해.
    그 다음에 <지역>을 물어봐야 해. 답변은 광역시나 도 단위로 나와야 해.(예: 서울, 인천, 경기도, 충청남도)
    만약 사용자가 광역시나 도 단위로 대답하지 않았거나 대한민국의 지역이 아니라면 다시 물어봐야해.
    그 다음에 일자리, 주거, 참여/권리, 복지/문화, 교육 정책 중 어느 <분야>인지 물어봐야 해.
    그런 다음 <분야>에 맞게 <세분류>를 물어봐야해. <세분류>는 중복해서 선택할 수 있어.
    각 <분야>에 속한 <세분류>는 아래와 같아.
    일자리-농어촌, 취업, 창업, 중소기업, 취업지원, 여성, 자격증 
    주거-주택, 자금지원, 금융
    복지/문화-복지, 문화, 건강, 행사, 임신/출산,
    참여/권리-참여, 권리, 법률상담, 커뮤니티, 지역발전
    교육-인재양성, 대학생, 장학금, 자기계발
    사용자가 <세분류>를 선택하지 않고 전부 알려달라고 하면 '전체'라고 해석해야 해.

    그런 다음 나이, 지역, 분야, 세분류 조건에 모두 해당하는 정책을 찾기위한 프롬프트를 아래 규칙에 맞게 말하해야 해.
    <>안의 값을 제외하고는 한글자도 틀리지 않고 정확하게 예시 그대로 말해야해.

    -규칙-
    1. 답변은 무조건 '<지역>에 거주하는 <나이>세를 위한 <분야> 정책을 찾아드릴게요(<세분류>). 잠시만 기다려주세요'의 형식으로 한다.
    2. <지역>에서 경남, 경북, 전남, 전북, 충남, 충북은 각각 경상남도, 경상북도, 전라남도, 전라북도, 충청남도, 충청북도로 풀어서 말한다.
    3. <세분류>가 여러개면 띄어쓰기 없이 '&'로 연결해서 출력한다.
    4. 임신/출산, 복지/문화, 참여/권리는 '임신/출산', '복지/문화', '참여/권리'로 출력한다.
    5. 규칙을 반드시 지킨다.

    -예시-
    user : '스무살', '서울', '교육', '대학생'
    you : '서울시에 거주하는 20세를 위한 교육 정책을 찾아드릴게요(대학생). 잠시만 기다려주세요.'
                                                               
    user : '31살입니다', '경기도에 살아요', '주거', '주택이랑 금융'                                                            
    you : '경기도에 거주하는 31세를 위한 주거 정책을 찾아드릴게요(주택&금융). 잠시만 기다려주세요.'
                                                               
    user : '28', '충남', '복지 문화', '전체'                                                       
    you : '충청남도에 거주하는 28세를 위한 복지/문화 정책을 찾아드릴게요(전체). 잠시만 기다려주세요.'
                                                               
    user: '19살', '인천시', '일자리정책', '취업 창업 중소기업'
    you : '인천시에 거주하는 19세를 위한 일자리 정책을 찾아드릴게요(취업&창업&중소기업). 잠시만 기다려주세요.'
                                                               
    """)

    # 2) ChatPromptTemplate 템플릿 정의
    prompt = ChatPromptTemplate.from_messages([
        system_message,                                              # 역할부여
        MessagesPlaceholder(variable_name="history"),           # 메모리 저장소 설정. ConversationBufferMemory의 memory_key 와 동일하게 설정
        HumanMessagePromptTemplate.from_template("{input}"),   # 사용자 메시지 injection
    ])

    # 3) LLM 모델 정의
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

    memory = ConversationBufferMemory(k=k,
                                      ai_prefix="검색 AI",
                                      human_prefix="사용자",
                                      return_messages=True)

    # 4) LLMChain 정의
    conversation = ConversationChain(
        llm=llm,       # LLM
        prompt=prompt, # Prompt
        verbose=True,  # True 로 설정시 로그 출력
        memory=memory  # 메모리
    )

    return conversation


def submit():
    user_input = st.session_state.user_input
    st.session_state.user_input = ''
    if (len(user_input) > 1):
        conversation = st.session_state[conversation_key]
        conversation.predict(input=user_input)


def main():
    st.set_page_config(page_title="Conversation Buffer Window Memory", page_icon=":robot:")
    st.title("정책 검색 서비스 : 정채기🔎")
    st.subheader("당신을 위한 맞춤 정책을 검색하고 싶다면 <정채기>한테 '안녕?'이라고 인사해주세요!")

    placeholder = st.empty()

    if st.button("안녕?"):
        user_input = "안녕?"
        conversation = st.session_state[conversation_key]
        conversation.predict(input=user_input)

    if conversation_key not in st.session_state:
        st.session_state[conversation_key] = getConversation()

    conversation = st.session_state[conversation_key]

    with placeholder.container():
        for index, msg in enumerate(conversation.memory.chat_memory.messages):
            if msg.type == human_message_key:
                message(msg.content, is_user=True, key=f"msg{index}")
            else:
                message(msg.content, key=f"msg{index}")


    st.text_input(label="Enter your message", placeholder="Send a message", key="user_input", on_change=submit)

    
if __name__ == '__main__':
    main()