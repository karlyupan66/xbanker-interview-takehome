import re
import uuid
import time
import logging

from contextlib import asynccontextmanager
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.schema import ChatMessage
from fastapi.responses import HTMLResponse

# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_TXT_SYS = "prompt_template_system.txt"
PROMPT_TEMPLATE_TXT_USER = "prompt_template_user.txt"

API_TYPE = "openai"
OPENAI_API_BASE = "https://api.openai.com/v1/"
OPENAI_CHAT_API_KEY = "your key"
OPENAI_CHAT_MODEL = "gpt-3.5-turbo"

PORT = 8080
model = None
prompt = None
chain = None


# 定义Message类
class Message(BaseModel):
    role: str
    content: str


# 定义ChatCompletionRequest类
class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    userId: Optional[str] = None
    conversationId: Optional[str] = None


# 定义ChatCompletionResponseChoice类
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


# 定义ChatCompletionResponse类
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    system_fingerprint: Optional[str] = None


# 获取历史对话
def get_session_history(user_id: str, conversation_id: str):
    history = SQLChatMessageHistory(f"{user_id}--{conversation_id}", "sqlite:///memory.db")
    all_messages = history.messages
    recent_messages_content = [message.content for message in all_messages[-10:]]
    history.clear()
    for content in recent_messages_content:
        message = ChatMessage(content=content, role="user")
        history.add_message(message)
    logger.info(f"历史对话记忆内容: {history}\n")
    return history


# 获取prompt在chain中传递的prompt最终的内容
def getPrompt(prompt):
    logger.info(f"最后给到LLM的prompt的内容: {prompt}")
    return prompt


def format_response(response):
    paragraphs = re.split(r'\n{2,}', response)
    formatted_paragraphs = []
    for para in paragraphs:
        if '```' in para:
            parts = para.split('```')
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            para = ''.join(parts)
        else:
            para = para.replace('. ', '.\n')
        formatted_paragraphs.append(para.strip())
    return '\n\n'.join(formatted_paragraphs)



# 初始化向量数据库和文档检索器
def initialize_vector_store():
    loader = TextLoader("financial_documents.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=OPENAI_CHAT_API_KEY))
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    return retriever



@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, prompt, chain, API_TYPE, PROMPT_TEMPLATE_TXT_SYS, PROMPT_TEMPLATE_TXT_USER, with_message_history
    global OPENAI_API_BASE, OPENAI_CHAT_API_KEY, OPENAI_CHAT_MODEL
    try:
        logger.info("正在初始化模型、提取prompt模版、定义chain...")

        model = ChatOpenAI(
            base_url=OPENAI_API_BASE,
            api_key=OPENAI_CHAT_API_KEY,
            model=OPENAI_CHAT_MODEL,
            temperature=0,
        )
        # 系统消息模板
        prompt_template_system = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT_SYS)
        prompt_template_user = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT_USER)

        # 创建 Prompt 模板
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template_system.template),
                MessagesPlaceholder(variable_name="history"),
                ("human", prompt_template_user.template)
            ]
        )


        # 定义chain
        chain = prompt | getPrompt | model
        with_message_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="query",
            docs_messages_key="retrieved_docs",
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="User ID",
                    description="Unique identifier for the user.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="conversation_id",
                    annotation=str,
                    name="Conversation ID",
                    description="Unique identifier for the conversation.",
                    default="",
                    is_shared=True,
                ),
            ],
        )



        logger.info("初始化完成")
    except Exception as e:
        logger.error(f"初始化过程中出错: {str(e)}")
        raise

    yield
    logger.info("正在关闭...")


app = FastAPI(lifespan=lifespan)


# POST请求接口，与大模型进行知识问答
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):

    # 申明引用全局变量，在函数中被初始化，并在整个应用中使用
    if not model or not prompt or not chain:
        logger.error("服务未初始化")
        raise HTTPException(status_code=500, detail="服务未初始化")

    try:
        logger.info(f"收到聊天完成请求: {request}")
        query_prompt = request.messages[-1].content
        logger.info(f"用户问题是: {query_prompt}")

        # 执行文档检索
        retriever = initialize_vector_store()
        relevant_docs = retriever.get_relevant_documents(query_prompt)
        messages = []
        for doc in relevant_docs:
            messages.append(Message(role="system", content=doc.page_content))

        # 执行综合 Chain
        result = with_message_history.invoke({"query": query_prompt, "retrieved_docs": messages},
                                             config={"configurable": {"user_id": request.userId,
                                                                      "conversation_id": request.conversationId}})

        # 格式化输出
        formatted_response = str(format_response(result.content))
        logger.info(f"格式化的搜索结果: {formatted_response}")

        response = ChatCompletionResponse(
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=formatted_response),
                    finish_reason="stop"
                )
            ]
        )
        logger.info(f"发送响应内容: \n{response}")
        return JSONResponse(content=response.model_dump())

    except Exception as e:
        logger.error(f"处理聊天完成时出错:\n\n {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as file:
        return HTMLResponse(file.read())




if __name__ == "__main__":
    logger.info(f"在端口 {PORT} 上启动服务器")
    # uvicorn是一个用于运行ASGI应用的轻量级、超快速的ASGI服务器实现
    # 用于部署基于FastAPI框架的异步PythonWeb应用程序
    uvicorn.run(app, host="127.0.0.1", port=PORT)





