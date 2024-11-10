# ChatGPT 知识问答 API

本项目是一个基于 OpenAI GPT-3.5-turbo 和 Langchain 的知识问答服务，通过与文档检索（使用 FAISS 向量存储）结合，能够在会话过程中提供更加智能和精确的答案。用户可以通过该服务进行查询，模型会根据历史对话和相关文档生成回答。

## 特性

- **带记忆的聊天机器人**：该服务可以记住之前的对话历史，并在后续交互中加以利用。
- **文档检索**：通过检索相关文档来增强模型回答的准确性。
- **文本格式化**：将模型的回答进行格式化，按段落结构进行展示。
- **基于 FastAPI 的 Web 服务**：提供一个 REST API，便于与其他应用程序集成。

## 需求

- **Python 3.7+**
- **FastAPI** - 用于构建 API 的 Web 框架
- **Uvicorn** - 用于服务应用的 ASGI 服务器
- **Langchain** - 用于与语言模型交互和管理文档加载的框架
- **OpenAI GPT-3.5-turbo** - OpenAI 提供的语言模型
- **FAISS** - 高效的相似度检索库（用于文档检索）
- **SQLite** - 用于存储聊天历史的数据库
- **Pydantic** - 数据验证和配置管理
- **Logging** - 用于跟踪请求进度和调试问题

## 安装与配置

### 1. 克隆代码仓库

```
git clone <repository-url>
cd <repository-directory>
```

### 2. 安装依赖
```
pip install -r requirements.txt
```

### 3. 设置 OpenAI API 密钥
确保你有一个有效的 OpenAI API 密钥。可以将 OPENAI_CHAT_API_KEY 替换为你自己的密钥，或者将其设置为环境变量：

```
export OPENAI_CHAT_API_KEY="your-api-key"
```
### 4. 配置文档
该服务从名为 financial_documents.txt 的文件中检索文档。请确保该文件包含你希望检索的相关知识。可以将文件放在工作目录下，或者修改代码中的路径。

### 5. 启动服务器
运行以下命令启动 FastAPI 服务器：

```
uvicorn app:app --host 127.0.0.1 --port 8080
```

默认情况下，服务器会在 http://127.0.0.1:8080 启动。

### 6. 访问 Web 界面
启动服务器后，你可以通过浏览器访问 Web 界面（如果存在 index.html 文件）：
```
http://127.0.0.1:8080/
```

## 代码概述

### `chat.py`

这是应用的主要文件，暴露了聊天 API 接口。关键部分包括：

- **全局变量**：在应用启动时初始化 `model`、`prompt` 和 `chain` 等变量。
- **`chat_completions` 端点**：该端点接受 POST 请求，处理聊天查询，通过检索相关文档和生成的模型响应进行回答。
- **`initialize_vector_store`**：初始化 FAISS 向量存储，从指定的文件中检索相关文档。
- **`get_session_history`**：根据用户和会话 ID 检索和管理聊天历史。
- **`format_response`**：格式化模型的输出，使其按照段落进行结构化展示。
- **`lifespan`**：管理应用生命周期，初始化模型、提示和链条（chain）。

### 其他关键文件

- **`prompt_template_system.txt`**：系统消息模板，提供给助手的背景信息。
- **`prompt_template_user.txt`**：用户消息模板，定义用户查询的格式。
- **`financial_documents.txt`**：包含用于文档检索的文本文件。

