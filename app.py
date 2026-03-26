import streamlit as st
from src.agent import agent, agent_rag, agent_sql, agent_python

print(agent_sql.tools)

st.set_page_config(
    page_title="LLM Агент",
    layout="wide"
)

st.title("Выбор агента: SQL / Retrieve / Python")
st.markdown("Введите запрос, и агент выбранного типа выполнит его.")


def call_agent(agent_type, user_input):
    """
    Вызывает соответствующего агента с текстом user_input.
    """
    global agent
    agent_map = {
        "sql": agent_sql,
        "retrieve": agent_rag,
        "python": agent_python
    }
    agent = agent_map[agent_type]
    try:
        response = agent.invoke(user_input)['output']
    except Exception as e:
        response = f"Ошибка при вызове агента: {e}"
    return response


# ------------------------------------------------------------
# 4. ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ СЕССИИ
# ------------------------------------------------------------
if "history" not in st.session_state:
    # История хранит список словарей: {"role": "user"/"assistant", "content": str}
    st.session_state.history = []

if "last_interaction" not in st.session_state:
    # Для быстрого доступа к последнему вопросу/ответу
    st.session_state.last_interaction = {"user": "", "assistant": ""}

if "agent_type" not in st.session_state:
    st.session_state.agent_type = "sql"


with st.sidebar:
    st.header("Настройки")

    # Выбор типа агента
    agent_type = st.radio(
        "Тип агента:",
        options=["sql", "retrieve", "python"],
        index=["sql", "retrieve", "python"].index(st.session_state.agent_type),
        key="agent_type_radio"
    )
    st.session_state.agent_type = agent_type

    st.markdown("---")
    if st.button("Очистить историю"):
        st.session_state.history = []
        st.session_state.last_interaction = {"user": "", "assistant": ""}
        st.rerun()

    st.markdown("---")
    st.markdown("### Инструкция")
    st.markdown(
        "1. Выберите тип агента.\n"
        "2. Введите запрос на вкладке **Чат**.\n"
        "3. Ответ появится сразу, а история сохранится на вкладке **История**.\n"
        "4. Агент **не получает предыдущие сообщения**, только текущий запрос."
    )


tab_chat, tab_history = st.tabs(["💬 Чат", "📜 История"])


with tab_chat:
    # Отображение последнего диалога (если есть)
    if st.session_state.last_interaction["user"]:
        with st.chat_message("user"):
            st.markdown(st.session_state.last_interaction["user"])
        with st.chat_message("assistant"):
            st.markdown(st.session_state.last_interaction["assistant"])

    # Поле ввода нового сообщения
    if prompt := st.chat_input("Введите запрос..."):
        # Сохраняем вопрос пользователя
        st.session_state.last_interaction["user"] = prompt
        st.session_state.history.append({"role": "user", "content": prompt})

        # Вызываем агента
        with st.chat_message("assistant"):
            with st.spinner(f"Агент ({st.session_state.agent_type}) думает..."):
                response = call_agent(st.session_state.agent_type, prompt)
            st.markdown(response)

        # Сохраняем ответ
        st.session_state.last_interaction["assistant"] = response
        st.session_state.history.append({"role": "assistant", "content": response})

        # Перезапускаем скрипт, чтобы обновить интерфейс
        st.rerun()

with tab_history:
    if not st.session_state.history:
        st.info("История пуста. Начните общение на вкладке Чат.")
    else:
        # Отображаем все сообщения в виде переписки
        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
