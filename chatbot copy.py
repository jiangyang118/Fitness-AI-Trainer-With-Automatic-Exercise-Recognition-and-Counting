import os
from dotenv import load_dotenv
import streamlit as st
from typing import Literal
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# 加载环境变量
load_dotenv()
#废弃，目前openai没有调通
@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

def build_llm():
    """构建LLM模型实例并处理初始化错误"""
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    api_key  = os.getenv("OPENAI_API_KEY", "").strip() or None
    # 统一用OPENAI_MODEL；若没配，按是否接DeepSeek选择默认值
    model = os.getenv("OPENAI_MODEL") or ("deepseek-chat" if base_url else "gpt-3.5-turbo")
    
    if not api_key:
        return None, "未检测到OPENAI_API_KEY，聊天已禁用。"
    
    try:
        llm = ChatOpenAI(
            model=model, 
            temperature=0.2,
            base_url=base_url, 
            api_key=api_key
        )
        return llm, None
    except Exception as e:
        return None, f"LLM初始化失败：{str(e)}"

def initialize_session_state(llm):
    """初始化会话状态，确保只初始化一次"""
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "history" not in st.session_state:
        st.session_state.history = []
    if "conversation" not in st.session_state and llm:
        # 初始化对话记忆
        conversation_memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="history"
        )
        # 设置AI角色：健身教练
        conversation_memory.save_context(
            {"human": "系统提示"}, 
            {"ai": "你是嵌入在健身应用中的AI教练，擅长指导家庭锻炼动作，能分类和计数重复动作，用专业且易懂的语言提供健身建议。"}
        )
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=conversation_memory,
        )

def on_click_callback():
    """处理消息提交逻辑"""
    human_prompt = st.session_state.get('human_prompt', '').strip()
    if human_prompt and "conversation" in st.session_state:
        # 获取AI响应
        llm_response = st.session_state.conversation.run(human_prompt)
        # 更新对话历史
        st.session_state.history.append(Message("human", human_prompt))
        st.session_state.history.append(Message("ai", llm_response))
        # 简单的token计数（实际应用需用tiktoken）
        st.session_state.token_count += len(human_prompt.split()) + len(llm_response.split())
        # 清空输入框
        st.session_state.human_prompt = ""

def chat_ui(context: str = ""):
    """聊天界面渲染"""
    # 构建LLM
    llm, err = build_llm()
    # 初始化会话状态
    initialize_session_state(llm)
    
    st.subheader("💬 AI 健身教练")
    if err or llm is None:
        st.info(err or "聊天功能未启用")
        return

    # 显示自定义CSS
    st.markdown("""
        <style>
            .chat-container {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            .chat-row {
                display: flex;
                width: 100%;
            }
            .row-reverse {
                flex-direction: row-reverse;
            }
            .chat-bubble {
                padding: 10px 15px;
                border-radius: 20px;
                margin-bottom: 5px;
                max-width: 70%;
                word-wrap: break-word;
            }
            .user-bubble {
                background-color: #007bff;
                color: white;
            }
            .ai-bubble {
                background-color: #f0f0f0;
                color: black;
            }
        </style>
    """, unsafe_allow_html=True)

    # 显示聊天历史
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for chat in st.session_state.history:
            bubble_class = "user-bubble" if chat.origin == "human" else "ai-bubble"
            row_class = "chat-row row-reverse" if chat.origin == "human" else "chat-row"
            st.markdown(f'''
                <div class="{row_class}">
                    <div class="chat-bubble {bubble_class}">
                        {chat.message}
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 输入表单
    with st.form("chat-form", clear_on_submit=True):
        user_input = st.text_input("请输入你的健身问题或需求...", 
                                 key="human_prompt",
                                 placeholder="例如：如何正确做俯卧撑？")
        submit_btn = st.form_submit_button("发送", on_click=on_click_callback)

    # 显示token计数（简单估算）
    st.caption(f"累计大约使用 {st.session_state.token_count} 个token")

if __name__ == "__main__":
    chat_ui()
