import chromadb

from langchain_community.llms import Ollama

from langchain.agents import AgentType
from langchain.agents import initialize_agent

from langchain.tools.retriever import create_retriever_tool

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langchain_experimental.tools import PythonREPLTool

from langchain.agents import initialize_agent
from langchain.tools import Tool
import re
import ast
import os

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
croma_path = os.getenv("CHROMA_DATA", "./chroma_data")
db_host = os.getenv("DB_HOST", "localhost")

if not ollama_host.startswith(('http://', 'https://')):
    ollama_host = 'http://' + ollama_host


collection_name = 'rosatom'
embed_model = 'yxchia/multilingual-e5-base:latest'
llm_model = 'qwen2.5:3b'
    

client = chromadb.PersistentClient(path=croma_path)
collection = client.get_or_create_collection(collection_name,
                                metadata={"hnsw:space": "cosine"})


llm = Ollama(model=llm_model, temperature=None, base_url=ollama_host)

embeddings = OllamaEmbeddings(model=embed_model, base_url=ollama_host)

vectorstore = Chroma(
    persist_directory=croma_path,
    collection_name=collection_name,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

rag_tool = create_retriever_tool(
    retriever,
    name="document_search",
    description="Инструмент помогает найти информацию во внутренней базе знаний."
)

db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://admin:pass@{db_host}/rearag",
    schema="hr"
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

sql_tools = toolkit.get_tools()



python_repl = PythonREPLTool()

def clean_python_code(code: str) -> str:
    # удаляем ```python ```py ```
    code = re.sub(r"```[a-zA-Z]*", "", code)
    code = code.replace("```", "")
    return code.strip()


def auto_print_last_expression(code: str):
    tree = ast.parse(code)

    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = tree.body[-1].value
        tree.body[-1] = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[last_expr],
                keywords=[]
            )
        )
        code = ast.unparse(tree)

    return code

def run_python(code: str):
    code = clean_python_code(code)
    code = auto_print_last_expression(code)
    return python_repl.run(code)

python_tool = Tool(
    name="Python_REPL",
    func=run_python,
    description="Выполняет Python код. Вход: чистый Python код. Чтобы увидеть результат, всегда делай print() результата в конце."
)


def clean_sql_input(input_text: str) -> str:
    """Удаляет маркдаун-блоки вида ```sql ... ``` и оставляет только SQL."""
    # Ищем блоки с тройными кавычками, возможно с указанием языка
    pattern = r"```(?:sql)?\s*(.*?)\s*```"
    match = re.search(pattern, input_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Если не нашли, возвращаем исходный текст

    pattern = r"```(?:py)?\s*(.*?)\s*```"
    match = re.search(pattern, input_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Если не нашли, возвращаем исходный текст
    
    return input_text.strip()

# Получаем оригинальные инструменты
sql_tools = toolkit.get_tools()

# Создаём новые инструменты с обёрткой
cleaned_sql_tools = []
for tool in sql_tools:
    if tool.name in ["sql_db_query", "sql_db_query_checker"]:
        # Обёртка для запросов
        def wrapped_run(query: str, _tool=tool):
            cleaned = clean_sql_input(query)
            return _tool.run(cleaned)
        new_tool = Tool(
            name=tool.name,
            func=wrapped_run,
            description=tool.description
        )
        cleaned_sql_tools.append(new_tool)
    else:
        cleaned_sql_tools.append(tool)


tools = [rag_tool, *cleaned_sql_tools, python_tool]


agent_kwargs = {
    "prefix": """
    
Ты — агент, работающий с базой знаний (документы) и SQL-базой данных.

Правила:
- Если вопрос можно решить поиском в документах — используй document_search.
- **Если document_search вернул достаточно информации для ответа — сразу дай ответ и остановись. Не используй SQL, если в этом нет необходимости.**
- Используйте SQL только тогда, когда вопрос явно требует данных из базы (например, подсчёты, списки, агрегация).
- SQL-запросы отправляй без обратных кавычек.
- Если нужно выполнить запрос к базе данных, всегда сначала узнавай структуру базы данных и точные имена столцов перед выполнением запросов, не нужно угадывать.
- Если тебе нужно выполнить вычисления, используй инструмент, который выполняет программы в Python.

### Правила работы с Python
1. Отправляй код без обратных кавычек ("```py"), только чистый Python.
2. Чтобы увидеть результат, всегда делай print() результата в конце.
3. Если ты не видишь результат выполнения кода, то скорее всего ты забыл применить print в конце. Добавь его в программу.

### ПРАВИЛА РАБОТЫ С SQL
1. **Сначала получи список таблиц** (sql_db_list_tables), чтобы понять, какие таблицы существуют.
2. **Затем посмотри структуру нужной таблицы** (sql_db_schema), чтобы узнать имена столбцов и типы данных. 
   - Инструмент sql_db_schema возвращает структуру + несколько примеров строк.
   - **Важно**: примеры строк — это только образец, они НЕ показывают все данные. 
     Не делай выводов о количестве записей, суммах и т.п. на основе этих примеров.
3. **Для точных ответов на вопросы о количестве, сумме, среднем и т.д. всегда составляй и выполняй SQL-запрос с агрегатными функциями** (COUNT, SUM, AVG) через инструмент sql_db_query.
   - Например: `SELECT COUNT(*) FROM table` — чтобы узнать число записей.
   - `SELECT SUM(column) FROM table WHERE condition` — для суммы.
4. Не угадывай имена столбцов — всегда сначала получай структуру.
5. SQL-запросы отправляй без обратных кавычек ("```"), только чистый SQL.


ФОРМАТ ТВОЕГО ОТВЕТА (строго соблюдай один из двух вариантов):

=== Вариант 1:Нужен инструмент ===
Thought: (твои рассуждения на русском языке)
Action: (название инструмента: document_search или sql_db_query, или другой доступный)
Action Input: (входные данные для инструмента)

=== Вариант 2: Готов дать ответ ===
Thought: (твои рассуждения на русском языке)
Final Answer: (ответ пользователю на русском языке)

ВАЖНО:
- После "Thought:" обязательно должно идти либо "Action:" с "Action Input:", либо "Final Answer:".
- Не добавляй никакого другого текста после "Final Answer:".

Отвечай строго на русском языке.
"""
}

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=13,
    agent_kwargs=agent_kwargs
)


agent_kwargs_sql = {
    "prefix": """
    
Ты — агент, работающий с SQL-базой данных.

Правила:
- SQL-запросы отправляй без обратных кавычек.
- Если нужно выполнить запрос к базе данных, всегда сначала узнавай структуру базы данных и точные имена столцов перед выполнением запросов, не нужно угадывать.

### ПРАВИЛА РАБОТЫ С SQL
1. **Сначала получи список таблиц** (sql_db_list_tables), чтобы понять, какие таблицы существуют. Пиши только названия таблиц БЕЗ дополнительных атрибутов (table= или table_name:), только названия таблиц!
2. **Затем посмотри структуру нужной таблицы** (sql_db_schema), чтобы узнать имена столбцов и типы данных. 
   - Инструмент sql_db_schema возвращает структуру + несколько примеров строк.
   - **Важно**: примеры строк — это только образец, они НЕ показывают все данные. 
     Не делай выводов о количестве записей, суммах и т.п. на основе этих примеров.
3. **Для точных ответов на вопросы о количестве, сумме, среднем и т.д. всегда составляй и выполняй SQL-запрос с агрегатными функциями** (COUNT, SUM, AVG) через инструмент sql_db_query.
   - Например: `SELECT COUNT(*) FROM table` — чтобы узнать число записей.
   - `SELECT SUM(column) FROM table WHERE condition` — для суммы.
4. Не угадывай имена столбцов — всегда сначала получай структуру.
5. SQL-запросы отправляй без обратных кавычек ("```"), только чистый SQL.
6. Для выполнения конкретного запроса используй команду sql_db_query, а не sql_db_query_checker! 


ФОРМАТ ТВОЕГО ОТВЕТА (строго соблюдай один из двух вариантов):

=== Вариант 1: Нужен инструмент ===
Thought: (твои рассуждения на русском языке)
Action: (любой доступный: sql_db_query, sql_db_query_checker, sql_db_list_tables, sql_db_schema)
Action Input: (входные данные для инструмента)
НИКОГДА НЕ ПИШИ "Final Answer:" при использовании инструментов!

=== Вариант 2: Готов дать ответ ===
Thought: (твои рассуждения на русском языке)
Final Answer: (конечный ответ пользователю на русском языке). Давай Final Answer, только когда получил ответы от инструментов и даешь ответ пользователю.

ВАЖНО:
- После "Thought:" обязательно должно идти либо "Action:" с "Action Input:", либо "Final Answer:".
- Не добавляй никакого другого текста после "Final Answer:".

Отвечай строго на русском языке.
"""
}

agent_sql = initialize_agent(
    [*cleaned_sql_tools],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=13,
    agent_kwargs=agent_kwargs_sql
)

agent_kwargs_rag = {
    "prefix": """
    
Ты — агент, работающий с базой знаний (документы).

ФОРМАТ ТВОЕГО ОТВЕТА (строго соблюдай один из двух вариантов):

=== Вариант 1:Нужен инструмент ===
Thought: (твои рассуждения на русском языке)
Action: (название инструмента: document_search или sql_db_query, или другой доступный)
Action Input: (входные данные для инструмента)

=== Вариант 2: Готов дать ответ ===
Thought: (твои рассуждения на русском языке)
Final Answer: (ответ пользователю на русском языке)

ВАЖНО:
- После "Thought:" обязательно должно идти либо "Action:" с "Action Input:", либо "Final Answer:".
- Не добавляй никакого другого текста после "Final Answer:".

Отвечай строго на русском языке.
"""
}

agent_rag = initialize_agent(
    [rag_tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=13,
    agent_kwargs=agent_kwargs_rag
)

agent_kwargs_pyth = {
    "prefix": """
    
Ты — агент, работающий с Python.

### Правила работы с Python
1. Отправляй код без обратных кавычек ("```py"), только чистый Python.
2. Чтобы увидеть результат, всегда делай print() результата в конце.
3. Если ты не видишь результат выполнения кода, то скорее всего ты забыл применить print в конце. Добавь его в программу.


ФОРМАТ ТВОЕГО ОТВЕТА (строго соблюдай один из двух вариантов):

=== Вариант 1:Нужен инструмент ===
Thought: (твои рассуждения на русском языке)
Action: (название инструмента: document_search или sql_db_query, или другой доступный)
Action Input: (входные данные для инструмента)

=== Вариант 2: Готов дать ответ ===
Thought: (твои рассуждения на русском языке)
Final Answer: (ответ пользователю на русском языке)

ВАЖНО:
- После "Thought:" обязательно должно идти либо "Action:" с "Action Input:", либо "Final Answer:".
- Не добавляй никакого другого текста после "Final Answer:".

Отвечай строго на русском языке.
"""
}

agent_python = initialize_agent(
    [python_tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=13,
    agent_kwargs=agent_kwargs_pyth
)



