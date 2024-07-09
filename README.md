# paper_summarization 🦜️🔗
This work for paper summarization.

LLM model:
- Llama 3 🦙
- GPT-4o
- GPT-4-turbo
- GPT-3.5-turbo

## How
1. Scrape Data
   
![image](https://github.com/JunukCha/paper_summarization/assets/92254092/17eb7570-c460-43c3-868d-974190d2ae2e)

Select the conference and year, and type your subject of interest (e.g., "hand").

![image](https://github.com/JunukCha/paper_summarization/assets/92254092/a0bd4d5c-cb84-46f1-9d40-4e7b867f9912)

Click the 'Scrape Data' button on the left side. Then, the paper list will be generated.

![image](https://github.com/JunukCha/paper_summarization/assets/92254092/b8461c72-637a-49d6-9c56-1b6ae3f85d50)

2. Save to Excel
Click the 'Save to Excel' button to download the dataframe of the paper list as an Excel file.
   
3. Save PDFs
Click the 'Save PDFs' button to download the PDF files of the papers in each folder.

4. Save Supps
Click the 'Save Supps' button to download the supplementary data of the papers in each folder.
   
5. Summary
Click the 'Summary' button to generate a markdown file and a Word file (docx) in each folder, containing the summary content.

Additional
1. If you use OPENAI's LLMs, then please type your API key.
2. You can add or remove the supplement PDF file for summary.
   

## Install
`source scripts/install.sh`

## Download LLama3
```
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
```

Please refer to [ChatOllama](https://python.langchain.com/v0.2/docs/integrations/chat/ollama/) for more details.

## Run
`source scripts/run.sh`

## Question Prompts
You can change the question prompts in prompt.py. I borrowed them from [Yunseong](https://github.com/yunseongcho/chatgpt_paper_review).
