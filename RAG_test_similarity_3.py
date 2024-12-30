from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import MergedDataLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain.prompts import MessagesPlaceholder
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableWithMessageHistory

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredExcelLoader #若有問題 改langchain_community回langchain
from langchain.prompts import MessagesPlaceholder


import numpy as np
import torch

import time
from tqdm import tqdm
# for STT
import pyaudio
import speech_recognition as sr
# for TTS
import pyttsx3

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings
warnings.filterwarnings("ignore")

# 配置參數
CONFIG = {
    "chunk_size": 300,
    "chunk_overlap": 50,
    "retriever_k": 2,
    "llm_model": "cwchang/llama3-taide-lx-8b-chat-alpha1",
    "embedding_model": "bge-m3",
    "vector_db_path": "./vectordb",
    "vector_db_index": "./vectordb/index.faiss",
    "data_folder_path": "", 
}

store = {}
def get_session_history(session_id:str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 初始化語言模型
def initialize_llm():
    # 初始化Ollama語言模型
    return OllamaLLM(model = CONFIG["llm_model"])


# 加載文檔並分割
def load_and_split_documents():
    # 載入並分割文件
    # pdf_files = [os.path.join(CONFIG["data_folder_path"], file) for file in os.listdir(CONFIG["data_folder_path"]) if file.endswith('.pdf')]
    # loaders = [PyPDFLoader(file) for file in pdf_files]
    # excel_files = [os.path.join(CONFIG["data_folder_path"], file) for file in os.listdir(CONFIG["data_folder_path"]) if file.endswith('.xlsx')]
    # loaders = [UnstructuredExcelLoader(file) for file in excel_files]
    txt_files = [os.path.join(CONFIG["data_folder_path"], file) for file in os.listdir(CONFIG["data_folder_path"]) if file.endswith('.txt')]
    loaders = [TextLoader(file, encoding='utf-8') for file in txt_files]

    # 加載並分割文檔
    docs = MergedDataLoader(loaders=loaders).load_and_split()

    # 查看加載的文檔數量
    # print(f"Loaded {len(docs)} documents.")


    # 設定文本分割器，chunk_size是分割的大小，chunk_overlap是重疊的部分
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CONFIG["chunk_size"], chunk_overlap=CONFIG["chunk_overlap"])
    return text_splitter.split_documents(docs)  # 將文件分割成更小的部分

# 初始化向量資料庫
def initialize_vector_db(documents):
    # 初始化向量的嵌入模型
    embeddings = OllamaEmbeddings(model=CONFIG["embedding_model"])

    # 添加進度條，並在文檔嵌入時顯示進度
    print("正在生成向量嵌入並建立 FAISS 資料庫...")
    start_time = time.time()

    # 使用FAISS建立向量資料庫
    if not os.path.exists(CONFIG["vector_db_index"]):
        vectordb = FAISS.from_documents(documents, embeddings)
        vectordb.save_local(CONFIG["vector_db_path"])
    else:
        vectordb = FAISS.load_local(CONFIG["vector_db_path"], embeddings, allow_dangerous_deserialization=True)

    end_time = time.time()
    print(f"向量資料庫建立完成，總耗時: {end_time - start_time:.2f} 秒，保存路徑: ./vectordb")

    return vectordb

# 設定問答鏈
def setup_chain(vectordb): 

    # 將向量資料庫設為檢索器
    retriever = vectordb.as_retriever(search_kwargs={"k": CONFIG["retriever_k"]})


    # 設定提示模板，將系統和使用者的提示組合
    prompt = ChatPromptTemplate.from_messages([
        # ('system', 'Answer the user\'s questions in Chinese, based on the context provided below:\n\n{context}'),
        # ('user', 'Question: {input}'),
        ("system", "你是一個問答小幫手，樂於以台灣人的立場幫助使用者，請用繁體中文回答問題。請只依我提供的檔案內容作回答，如果超出檔案內容回答不知道。回覆時請簡潔，不要長篇大論，也不要加上自己的見解。{context}"),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}")
    ])

    # 創建文件鏈，將llm和提示模板結合
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 創建檢索鏈，將檢索器和文件鏈結合
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversational_rag_chain
    


if __name__ == "__main__":
    print("請稍等...")
    
    # 初始化
    llm = initialize_llm()
    documents = load_and_split_documents()
    vectordb = initialize_vector_db(documents)
    conversational_rag_chain = setup_chain(vectordb)


    # 初始化語音合成器
    engine = pyttsx3.init()

    # 創建識別器
    r = sr.Recognizer()

    # 使用麥克風錄製音訊
    with sr.Microphone() as source:
        print("請開始說話，按 Ctrl+C 停止。")

        # 調整音量以獲取更好的結果
        r.adjust_for_ambient_noise(source)

        while True:
            try:
                print("聆聽中...")
                audio = r.listen(source)
                print("正在辨識...")

                # 使用 Google 語音辨識 API 辨識中文
                text = r.recognize_google(audio, language='zh-TW')
                print("你說的內容是: " + text)
                
                #llama產生回答
                if text.lower() != 'bye':
                    time_start = time.time()

                    # 依據使用者的問題，列出尋找到的向量化段落，並依照相似度排序
                    docs_with_scores = vectordb.similarity_search_with_score(text, k=2)  # 檢索兩個結果，包含分數
                    
                    # 顯示檢索結果和相似度值
                    for i, (doc, score) in enumerate(docs_with_scores):
                        print(f"\n第 {i+1} 個相似段落：\n{doc.page_content}")
                        print(f"相似度距離分數: {score:.4f}") 

                    response = conversational_rag_chain.invoke({
                        'input': text,
                    },config={
                        'configurable': {'session_id': 'test123'}
                    })
                    print('\n\n' + response['answer'])
                    time_end = time.time()
                    print('\ntime cost: ', time_end - time_start,' s\n')

                # 語音回應
                response_with_voice = f"{response['answer']}"
                engine.say(response_with_voice)
                engine.runAndWait()

            except sr.UnknownValueError:
                print("抱歉，我無法理解你說的內容。")
            except sr.RequestError as e:
                print(f"無法連接到語音辨識服務; {e}")
            except KeyboardInterrupt:
                print("停止辨識。")
                break
