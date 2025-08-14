import os
import time
from typing import List, Dict, Optional

import requests
import streamlit as st
from dotenv import load_dotenv
import json
import itertools

# 载入 .env（支持本地开发）
load_dotenv()

# ---------- DeepSeek 最小客户端 ----------

def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    return v if v and v.strip() else default

def build_deepseek():
    """
    仅使用 DeepSeek 官方兼容的 Chat Completions 接口。
    环境变量：
      - DEEPSEEK_API_KEY（必填，兼容 OPENAI_API_KEY）
      - DEEPSEEK_BASE_URL（可选，默认 https://api.deepseek.com/v1）
      - DEEPSEEK_MODEL（可选，默认 deepseek-chat）
    """
    api_key = _env("DEEPSEEK_API_KEY") or _env("OPENAI_API_KEY")
    base_url = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model = _env("DEEPSEEK_MODEL", "deepseek-chat")
    if not api_key:
        return None, "未检测到 DEEPSEEK_API_KEY（或 OPENAI_API_KEY），聊天已禁用。"

    def chat(messages: List[Dict[str, str]], temperature: float = 0.2):
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def chat_stream(messages: List[Dict[str, str]], temperature: float = 0.2):
        """
        流式输出：逐块返回content增量（兼容OpenAI风格SSE：data: {...}）。
        产出(str)：每次增量文本（可能为空字符串）。
        """
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        with requests.post(url, headers=headers, json=payload, timeout=300, stream=True) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data: "):
                    data_str = line[len("data: "):].strip()
                else:
                    # 兼容可能不带前缀的实现
                    data_str = line
                if data_str == "[DONE]":
                    break
                try:
                    evt = json.loads(data_str)
                except Exception:
                    continue
                try:
                    delta = evt["choices"][0].get("delta") or {}
                    text = delta.get("content", "")
                except Exception:
                    text = ""
                if text:
                    yield text

    return (chat, chat_stream), None

# ---------- Streamlit 对话 UI ----------

def chat_ui(context: str = ""):
    st.subheader("💬 AI 健身教练（DeepSeek）")

    (chat, chat_stream), err = build_deepseek()
    if err or chat is None:
        st.info(err or "聊天未启用")
        return

    # 统一消息状态：[{role, content}]
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 首次注入 system（含可选业务上下文）
    if context and (not st.session_state["messages"] or st.session_state["messages"][0].get("role") != "system"):
        st.session_state["messages"].insert(0, {
            "role": "system",
            "content": (
                "你是嵌入在健身应用中的AI教练，擅长指导家庭锻炼动作，"
                "能分类与计数重复动作，用专业且易懂的语言提供训练建议。"
                f"上下文：{context}"
            )
        })

    # 顶部操作
    cols = st.columns(2)
    with cols[0]:
        if st.button("🧹 清空会话", use_container_width=True):
            st.session_state["messages"] = st.session_state["messages"][:1]  # 保留 system
            st.rerun()
    with cols[1]:
        temperature = st.slider("Temperature（越低越稳）", 0.0, 1.0, 0.2, 0.1)

    # 回放历史（忽略 system）
    for m in st.session_state["messages"]:
        if m["role"] == "system":
            continue
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])

    # 输入
    prompt = st.chat_input("请输入你的健身问题或需求…")
    if prompt:
        user_msg = {"role": "user", "content": prompt}
        st.session_state["messages"].append(user_msg)
        with st.chat_message("user"):
            st.markdown(prompt)

        # 模型回复
        with st.chat_message("assistant"):
            with st.spinner("DeepSeek 思考中…"):
                start_time = time.time()
                text_placeholder = st.empty()        # 展示增量内容
                timer_placeholder = st.empty()       # 展示读秒
                content_buf = []
                try:
                    # 一边读流一边刷新内容与计时
                    for chunk in chat_stream(st.session_state["messages"], temperature=float(temperature)):
                        content_buf.append(chunk)
                        # 更新文本
                        text_placeholder.markdown("".join(content_buf))
                        # 更新计时（读秒）
                        elapsed = time.time() - start_time
                        timer_placeholder.caption(f"⏱ 正在生成：{elapsed:.1f} 秒")
                    # 结束后显示总耗时
                    total_elapsed = time.time() - start_time
                    timer_placeholder.caption(f"✅ 完成，耗时：{total_elapsed:.2f} 秒")
                    final_text = "".join(content_buf).strip()
                    if not final_text:
                        final_text = "_（无返回内容）_"
                    st.session_state["messages"].append({"role": "assistant", "content": final_text})
                except requests.HTTPError as e:
                    try:
                        err_detail = e.response.json()
                    except Exception:
                        err_detail = {"message": str(e)}
                    st.error(f"DeepSeek 流式调用错误：{err_detail}")
                    return
                except Exception as e:
                    st.error(f"DeepSeek 流式异常：{e}")
                    return

# 入口
if __name__ == "__main__":
    chat_ui()