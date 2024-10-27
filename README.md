# Your Pro Financial Agent

## 简介

Your Pro Financial Agent 是一个基于 Flask 和 LangChain 的智能金融助手，旨在帮助用户解答各种金融问题，包括投资策略、理财建议、市场分析等。该应用程序使用 OpenAI 的 GPT 模型来生成响应，并能够记录和回顾用户的对话历史。

## 功能

- **智能对话**：用户可以与金融助手进行自然语言对话，获取实时的金融建议和信息。
- **对话历史记录**：应用程序会记录用户的对话历史，用户可以随时查看之前的对话。
- **自我介绍**：在首次使用时，助手会提供自我介绍，帮助用户了解其功能。
- **总结与询问**：在对话结束时，助手会总结之前的对话并询问用户是否需要进一步的帮助。
- **清除聊天记录**：用户可以选择清除所有聊天记录，重新开始对话。

## 安装与使用

### 先决条件

确保您的系统上已安装以下软件：

- Python 3.7 或更高版本
- pip（Python 包管理器）

### 安装步骤

1. **克隆仓库**：

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **创建虚拟环境（可选）**：

   ```bash
   python -m venv venv
   source venv/bin/activate  # 在 Windows 上使用 venv\Scripts\activate
   ```

3. **安装依赖**：

   ```bash
   pip install -r requirements.txt
   ```

4. **设置环境变量**：

   创建一个 `.env` 文件，并添加以下内容：

   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

   请确保将 `your_openai_api_key` 和 `your_tavily_api_key` 替换为您自己的 API 密钥。

5. **运行应用程序**：

   ```bash
   python app.py
   ```

6. **访问应用程序**：

   打开浏览器并访问 `http://127.0.0.1:5000`。

## 使用说明

- 在聊天框中输入您的问题或请求，然后点击“发送”按钮或按 `Ctrl + Enter` 发送消息。
- 您可以随时查看之前的对话记录，助手会根据历史记录提供更相关的响应。
- 点击“清除聊天”按钮可以清除所有聊天记录并重新开始对话。

## 贡献

欢迎任何形式的贡献！如果您有建议或发现了错误，请提交问题或拉取请求。

## 许可证

该项目使用 MIT 许可证。有关详细信息，请参阅 [LICENSE](LICENSE) 文件。
