# Fullstack interview challenge

You are given two options below and you can choose either one to work on. Make sure to keep things simple. Overall it should not take more than four hours to complete.
The challenges contain both frontend and backend components with emphasis more on the backend side.

What we are looking for?
 - Fully functional and working code that we can run.
 - Code quality. We value readable and modular code
 - Tests. Treat this project as if it is meant to be deployed to production

Things to keep in mind:
- Use any frontend framework of your choice; the focus should be on frontend logic rather than style
  - Some frontend frameworks: (https://ant.design/, https://mui.com/material-ui/, https://preline.co/, etc...)
- Use any backendend languange and framework of your choice
  - Some backend frameworks: (fastapi, flask, ktor, etc..)
- Choose any database technologies you see fit

# Challenge

## Option 1: Portfolio Performance

Build a SPA application where portfolio managers can enter their client's assets portfolio distributions at some past date and see how much money would be worth today.

#### Example: 
- Client name: John Smith 
- Start Date: 2024-01-10
- Initial Balance: $500K
- Portfolio Allocation:
  - AAPL: 20%
  - GOOG: 80%

The app should show client's individual stock performance and overall portfolio return. Basic requirement is to show these data as a table. 
Optionally feel free to visualize the output as you see fit (Aggregate portfolio returns chart, Portfolio Allocation evolution over time, etc). 

Additionally the app should allow portfolio managers to view all previous request data, i.e. list of previous searchs PM initiated. 
You can display these past searches as a table where client name and time are shown, you can show the actual portfolio performance data upon click.

To obtain the historical returns you can use this free API https://www.worldtradingdata.com/

## Option 2: Financial assistant chatbot

Build a web-based chat application powered by LLM and agent. 

Requirements:
 - Your chatbot should utialize agent and general knowledge/reasoning of LLM to answer basic financial related questions
 - Persist all chat messages, use the database of your choice
 - The chatbot should have basic memory capability in order to carry coherent conversation

You can choose any Open AI models as the LLM of your choice. We will provide a test Open AI key for you to use in your development.
Feel free to directly use Open AI API calls or use any existing LLM frameworks like Langchain or Llamaindex.

For simplicity, your agent can utilize free search API provided by https://tavily.com/. 

