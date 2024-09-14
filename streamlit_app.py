import os
import streamlit as st
import openai
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI as LangOpenAI
from langchain.chains import RetrievalQA
import requests
import xml.etree.ElementTree as ET
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

# Pydantic 설정에서 임의 타입 허용하기
class CustomConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True

# 기존 코드에 적용
# 이 부분을 LangChain 코드와 통합하여 사용하세요.

# Streamlit을 위한 설정
st.set_page_config(page_title="Yam Yam Bot", 
                   page_icon=":bridge_at_night:", 
                   layout="wide")

st.title("# :rainbow[맛집추천 얌얌봇]")

openai_api_key = st.secrets["openai"]["api_key"]

# OpenAI API 키 설정을 위한 함수
def init_api():
    with open("chatgpt.env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

# XML 데이터를 가져오는 함수
def get_xml_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # 오류 발생 시 예외 처리
        return response.content  # XML 데이터를 반환
    except requests.exceptions.RequestException as e:
        print(f"API 호출 실패: {e}")
        return None

# XML 데이터를 파싱하는 함수
def parse_xml_data(xml_data):
    try:
        root = ET.fromstring(xml_data)  # XML 데이터를 파싱
        return ET.tostring(root, encoding='utf-8').decode('utf-8')  # XML 데이터를 문자열로 변환
    except ET.ParseError as e:
        print(f"XML 파싱 실패: {e}")
        return None

# LLM(대형 언어 모델)의 응답을 처리하는 함수
def process_llm_response(llm_response):
    st.write("**답변:**")
    st.write(llm_response['result'])

# Streamlit 사용자 입력
user_question = st.text_input("**먹고 싶은 음식 관련 맛집을 검색해보세요!**", "")

if st.button("답변 받기"):
    if user_question:
        # API 초기화 및 키 설정
        
        client = OpenAI(api_key=openai_api_key)

        # API URL로부터 XML 데이터 가져오기
        api_url = "https://openapi.gg.go.kr/Familyrstrt"
        xml_data = get_xml_data(api_url)
        if xml_data:
            parsed_data = parse_xml_data(xml_data)
            documents = parsed_data

            # 텍스트 분할기 설정
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_text(documents)

            # 텍스트 벡터화 및 벡터 데이터베이스 설정
            persist_directory = 'db'
            embedding = OpenAIEmbeddings()
            vectordb = Chroma.from_texts(texts=texts, embedding=embedding, persist_directory=persist_directory)

            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
            retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2})

            # OpenAI LLM을 사용하여 질문-답변 체인 설정
            qa_chain = RetrievalQA.from_chain_type(
                llm=LangOpenAI(),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )

            # 질문에 대한 답변 생성 및 처리
            response = qa_chain({"query": user_question})
            process_llm_response(response)
        else:
            st.error("XML 데이터를 가져오는 데 실패했습니다.")
    else:
        st.warning("질문을 입력해 주세요.")
