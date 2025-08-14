import os
from dotenv import load_dotenv
import streamlit as st
from typing import Literal
from dataclasses import dataclass
# LangChain核心依赖
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


#废弃，目前openai没有调通
# 修复HTTP头部编码问题（兼容中文且避免类型冲突）
import httpx
from httpx._models import _normalize_header_value

def patch_normalize_header_value(value, encoding=None):
    """修正头部值编码，优先保留字符串类型，避免bytes冲突"""
    if isinstance(value, bytes):
        return value  # 已为bytes类型直接返回
    if not isinstance(value, str):
        raise TypeError(f"Header value must be str or bytes, not {type(value).__name__}")
    # 优先用ASCII编码，失败时自动切换为UTF-8（解决中文问题）
    try:
        return value.encode(encoding or "ascii")
    except UnicodeEncodeError:
        return value.encode("utf-8")

# 应用补丁
httpx._models._normalize_header_value = patch_normalize_header_value

# 加载环境变量（从.env文件）
load_dotenv()

@dataclass
class Message:
    """聊天消息数据结构"""
    origin: Literal["human", "ai"]  # 消息来源：用户/AI
    message: str  # 消息内容

def build_llm(selected_model):
    """
    根据选择的模型构建LLM实例
    返回：(llm实例, 错误信息)
    """
    api_key = None
    base_url = None
    custom_headers = None  # 自定义HTTP头部（适配DeepSeek）
    
    if selected_model == "OpenAI":
        # OpenAI配置
        api_key = str(os.getenv("OPENAI_API_KEY", "").strip())  # 强制字符串类型
        base_url = str(os.getenv("OPENAI_BASE_URL", "").strip() or None)
        default_model = "gpt-3.5-turbo"
    elif selected_model == "DeepSeek":
        # DeepSeek配置（显式处理头部避免类型错误）
        api_key = str(os.getenv("DEEPSEEK_API_KEY", "").strip())
        base_url = str(os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip())
        default_model = "deepseek-chat"
        # 手动构建Authorization头部（解决DeepSeek的API密钥类型问题）
        if api_key:
            custom_headers = {"Authorization": f"Bearer {api_key}"}
    
    # 模型名称优先级：环境变量 > 默认值
    model = os.getenv(f"{selected_model.upper()}_MODEL") or default_model
    
    # 校验API密钥
    if not api_key:
        return None, f"未检测到{selected_model}的API密钥，请检查.env文件"
    
    try:
        # 初始化LLM（DeepSeek通过headers传递密钥，避免api_key参数类型冲突）
        llm = ChatOpenAI(
            model=model,
            temperature=0.2,  # 控制输出随机性（0.2较稳定）
            base_url=base_url,
            api_key=api_key if selected_model != "DeepSeek" else None,  # DeepSeek用headers
            default_headers=custom_headers  # 传递自定义头部
        )
        return llm, None
    except Exception as e:
        return None, f"{selected_model}初始化失败：{str(e)}"

def reset_session():
    """重置会话状态（模型切换时调用）"""
    for key in ["token_count", "history", "conversation", "current_model"]:
        if key in st.session_state:
            del st.session_state[key]

def initialize_session_state(llm, selected_model):
    """初始化会话状态，确保变量正确赋值"""
    # 模型切换时强制重置会话
    if "current_model" in st.session_state and st.session_state.current_model != selected_model:
        reset_session()
    
    st.session_state.current_model = selected_model  # 记录当前模型
    
    # 初始化基础状态（无依赖项）
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0  # 简单token计数
    if "history" not in st.session_state:
        st.session_state.history = []  # 聊天历史
    
    # 仅当LLM有效时，初始化对话链
    if llm is not None and "conversation" not in st.session_state:
        # 1. 创建记忆组件（替代过时的ConversationSummaryMemory）
        conversation_memory = ConversationSummaryBufferMemory(
            llm=llm,
            memory_key="history",  # 与prompt中的变量名对应
            return_messages=True,  # 返回Message对象而非字符串
            max_token_limit=1000  # 控制记忆长度（避免超限）
        )
        
        # 2. 定义对话提示模板（系统角色+历史+用户输入）
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是嵌入在健身应用中的AI教练，擅长指导家庭锻炼动作，能分类和计数重复动作，用专业且易懂的语言提供健身建议。"),
            MessagesPlaceholder(variable_name="history"),  # 对话历史
            ("human", "{input}")  # 用户输入
        ])
        
        # 3. 构建带历史的对话链（替代deprecated的ConversationChain）
        chain = prompt | llm  # LCEL语法：prompt -> llm
        st.session_state.conversation = RunnableWithMessageHistory(
            chain,
            lambda session_id: conversation_memory,  # 记忆管理函数
            input_messages_key="input",  # 用户输入的key
            history_messages_key="history"  # 历史消息的key
        )
    # 若LLM无效，清除对话链
    elif llm is None and "conversation" in st.session_state:
        del st.session_state.conversation

def on_click_callback():
    """消息提交回调函数（带完整错误处理）"""
    human_prompt = st.session_state.get('human_prompt', '').strip()
    
    # 基础校验
    if not human_prompt:
        st.warning("请输入内容后再发送")
        return
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.error("模型未初始化，请检查API密钥或重新选择模型")
        return
    
    try:
        # 调用AI生成回复（显示加载状态）
        with st.spinner("AI正在思考..."):
            # 调用对话链（需指定session_id区分会话）
            response = st.session_state.conversation.invoke(
                {"input": human_prompt},
                config={"configurable": {"session_id": "fitness_chat"}}  # 固定session_id
            )
            llm_response = response.content  # 提取回复内容
        
        # 更新聊天历史和token计数
        st.session_state.history.append(Message("human", human_prompt))
        st.session_state.history.append(Message("ai", llm_response))
        st.session_state.token_count += len(human_prompt.split()) + len(llm_response.split())
    
    except Exception as e:
        st.error(f"对话出错：{str(e)}")
    
    # 清空输入框
    st.session_state.human_prompt = ""

def chat_ui(context: str = ""):
    """渲染聊天界面"""
    st.subheader("💬 AI 健身教练")
    
    # 模型选择下拉框
    selected_model = st.selectbox(
        "选择大模型",
        ["OpenAI", "DeepSeek"],
        index=0,
        help="切换不同的AI模型进行对话（需配置对应API密钥）"
    )
    
    # 构建模型并初始化会话
    llm, err = build_llm(selected_model)
    initialize_session_state(llm, selected_model)
    
    # 显示错误信息（如API密钥缺失）
    if err or llm is None:
        st.info(err or "聊天功能未启用")
        return

    # 自定义聊天气泡样式
    st.markdown("""
        <style>
            .chat-container { display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px; }
            .chat-row { display: flex; width: 100%; }
            .row-reverse { flex-direction: row-reverse; }
            .chat-bubble { padding: 10px 15px; border-radius: 20px; max-width: 70%; word-wrap: break-word; }
            .user-bubble { background-color: #007bff; color: white; }
            .ai-bubble { background-color: #f0f0f0; color: black; }
        </style>
    """, unsafe_allow_html=True)
    
    # 显示聊天历史
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for chat in st.session_state.history:
            bubble_class = "user-bubble" if chat.origin == "human" else "ai-bubble"
            row_class = "chat-row row-reverse" if chat.origin == "human" else "chat-row"
            st.markdown(f'''
                <div class="{row_class}">
                    <div class="chat-bubble {bubble_class}">{chat.message}</div>
                </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 用户输入表单
    with st.form("chat-form", clear_on_submit=True):
        user_input = st.text_input(
            "请输入你的健身问题或需求...", 
            key="human_prompt",
            placeholder="例如：如何正确做俯卧撑？"
        )
        submit_btn = st.form_submit_button("发送", on_click=on_click_callback)

    # 显示token计数（简单估算）
    st.caption(f"累计大约使用 {st.session_state.token_count} 个token")

# 主函数入口
if __name__ == "__main__":
    chat_ui()