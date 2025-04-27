import streamlit as st
import requests

# ── Session state initialization ──────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.practice_name = ""
    st.session_state.user_name = ""
    st.session_state.messages = [
        {"sender": "bot", "content": "Welcome! What is your practice name?"}
    ]

st.set_page_config(page_title="Sikka Chatbot Assistant")

# ── Display past messages ─────────────────────────────────────────
for msg in st.session_state.messages:
    st.chat_message(msg["sender"]).markdown(msg["content"])

# ── Input handling ────────────────────────────────────────────────
if st.session_state.step == 0:
    # ask practice name
    practice = st.chat_input("Practice name…")
    if practice:
        st.session_state.practice_name = practice
        st.session_state.messages.append({"sender": "user", "content": practice})
        st.session_state.messages.append(
            {"sender": "bot", "content": "Great! What is your name?"}
        )
        st.session_state.step = 1

elif st.session_state.step == 1:
    # ask user name
    name = st.chat_input("Your name…")
    if name:
        st.session_state.user_name = name
        st.session_state.messages.append({"sender": "user", "content": name})
        st.session_state.messages.append(
            {"sender": "bot", "content": "How can I help you today?"}
        )
        st.session_state.step = 2

else:
    # main chat loop
    user_input = st.chat_input("Type your message…")
    if user_input:
        st.session_state.messages.append({"sender": "user", "content": user_input})
        # show a loading spinner
        with st.spinner("Sikka is typing…"):
            res = requests.post(
                "http://localhost:8000/chat",
                json={
                    "customer": st.session_state.practice_name,
                    "person": st.session_state.user_name,
                    "inquiry": user_input,
                },
                timeout=20,
            )
        if res.status_code == 200:
            bot_reply = res.json().get("response", "Sorry, something went wrong.")
        else:
            bot_reply = "Sorry, something went wrong."
        st.session_state.messages.append(
            {"sender": "bot", "content": bot_reply}
        )
        # Rerun so that messages appear immediately
        st.experimental_rerun()
