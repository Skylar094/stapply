import streamlit as st
import requests
import json

# --- 用户画像类别及特征词定义 (基本不变，描述可微调以更中性) ---
USER_PROFILE_DEFINITIONS = [
    {
        "name": "发育问题",
        "description": "涉及儿童生长发育、认知、语言、运动能力发展相关的咨询。",
        "keywords": ["迟缓", "发育", "障碍", "评估", "干预", "训练", "宝宝", "认知", "运动", "注意力", "疑似", "落后",
                     "语言", "能力", "行为", "抬头", "走路", "说话晚"]
    },
    {
        "name": "疾病影响问题",
        "description": "关于特定疾病（如阿斯伯格、自闭症谱系、多动症等）对生活学习的影响、药物、治疗方案相关的咨询。",
        "keywords": ["阿斯伯格", "自闭症", "谱系", "多动症", "注意力缺陷", "焦虑症", "情绪障碍", "治疗", "用药", "药物",
                     "副作用", "行为问题", "社交障碍", "康复", "诊断", "影响", "综合征", "症状"]
    },
    {
        "name": "生活状态问题",
        "description": "关于儿童在日常生活、学习、社交场景中的具体表现和适应情况的咨询。",
        "keywords": ["语言交流", "说话", "宝宝", "小朋友", "幼儿园", "学校", "沟通障碍", "互动", "社交", "表现", "适应",
                     "交往", "不合群", "生活自理"]
    },
    {
        "name": "检查问题",
        "description": "关于各类医学检查（特别是发育行为、心理情绪相关）的目的、流程、注意事项、结果解读等方面的咨询。",
        "keywords": ["检查", "治疗", "诊断", "评估", "量表", "结果", "医院", "项目", "费用", "筛查", "测试", "报告",
                     "看什么科"]
    },
    {
        "name": "情绪问题",
        "description": "涉及儿童和青少年情绪波动、心理压力、焦虑、抑郁、学习压力、适应困难等情绪心理方面的咨询。",
        "keywords": ["情绪", "阿斯伯格", "学校", "适应", "治疗", "学业", "情绪障碍", "诊断", "焦虑", "抑郁", "学业压力",
                     "心理", "行为", "压力", "哭闹", "发脾气", "紧张"]
    }
]

USER_PROFILE_CATEGORY_NAMES = [profile["name"] for profile in USER_PROFILE_DEFINITIONS]


# --- 核心API调用函数 (与上一版一致) ---
def call_siliconflow_api(api_key, model_name, messages_payload, max_tokens, temperature=0.7, top_p=0.7,
                         frequency_penalty=0.5):
    url = "https://api.siliconflow.cn/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages_payload,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        result = response.json()
        if result.get("choices") and len(result["choices"]) > 0:
            message = result["choices"][0].get("message")
            if message and message.get("content"):
                return message["content"].strip()
            else:
                return f"API 返回数据格式错误：缺少 'message' 或 'content'。\n原始返回: {json.dumps(result, indent=2, ensure_ascii=False)}"
        else:
            error_code = result.get("error", {}).get("code", "N/A")
            error_message = result.get("error", {}).get("message", "未知错误")
            if not error_message and isinstance(result.get("error"), str): error_message = result.get("error")
            if not error_message and isinstance(result.get("message"), str): error_message = result.get("message")
            return f"API 调用失败：错误码 '{error_code}', 信息: '{error_message}'.\n原始返回: {json.dumps(result, indent=2, ensure_ascii=False)}"
    except requests.exceptions.Timeout:
        return "API 请求超时，请稍后再试。"
    except requests.exceptions.HTTPError as http_err:
        error_body = ""
        try:
            error_body = http_err.response.json()
        except json.JSONDecodeError:
            error_body = http_err.response.text
        return f"API HTTP 错误: {http_err}\n响应内容: {json.dumps(error_body, indent=2, ensure_ascii=False) if isinstance(error_body, dict) else error_body}"
    except requests.exceptions.RequestException as e:
        return f"API 请求失败: {e}"
    except json.JSONDecodeError:
        return f"API 响应解析失败 (非JSON格式): {response.text if 'response' in locals() else '无响应内容'}"
    except Exception as e:
        return f"调用API时发生未知错误: {e}"


# --- 问题分类函数 (修改Prompt为医生视角) ---
def get_query_classification(api_key, user_query, model_name="Qwen/Qwen1.5-7B-Chat"):
    categories_prompt_text = "预定义咨询类别及其描述和典型特征词：\n\n"
    for i, profile in enumerate(USER_PROFILE_DEFINITIONS):
        categories_prompt_text += f"{i + 1}. 类别名称: {profile['name']}\n"
        categories_prompt_text += f"   - 描述: {profile['description']}\n"
        categories_prompt_text += f"   - 相关特征词: {', '.join(profile['keywords'])}\n\n"

    prompt_template = f"""你是一位专业的医疗辅助AI，正在协助医生对患者或家属的初步咨询进行分类。
请仔细阅读以下患者或家属提出的问题，并严格按照下面提供的预定义咨询类别及其相关的特征词，将该问题准确地归类。
你的任务是判断该咨询最符合哪个类别，以便医生快速把握咨询方向。
请只输出最匹配的类别名称，确保该名称与“预定义咨询类别及其描述和典型特征词”列表中提供的“类别名称”完全一致。不要添加任何额外的解释、编号或其它文字。

{categories_prompt_text}
患者或家属的咨询：“{user_query}”

最匹配的类别名称是：
"""
    messages = [{"role": "user", "content": prompt_template}]
    classification = call_siliconflow_api(api_key, model_name, messages, max_tokens=80, temperature=0.05)
    if classification in USER_PROFILE_CATEGORY_NAMES:
        return classification
    else:
        for cat_name in USER_PROFILE_CATEGORY_NAMES:
            if cat_name in classification:
                return cat_name
        return f"未能准确识别咨询类别 (模型返回: {classification})"


# --- 获取回答的函数 (修改Prompt为医生视角，并接收分类结果) ---
def get_answer_from_llm(api_key, user_query, classification_result, knowledge_base_content,
                        model_name="Qwen/Qwen1.5-32B-Chat"):
    prompt_template = f"""你是一位专业的医疗辅助AI，正在协助医生对一个关于自闭症谱系障碍（ASD）的初步咨询进行分析。
患者或家属的原始咨询内容初步归类为：“{classification_result}”。
患者或家属的原始咨询具体内容是：“{user_query}”

请仔细阅读并参考以下关于自闭症（ASD）的专业知识库：
--- 专业知识库 ---
{knowledge_base_content}
--- 专业知识库结束 ---

基于患者或家属的咨询以及上述专业知识库，请为医生提供以下辅助信息：
1.  **咨询要点总结**: 简要总结患者或家属咨询的核心内容和主要关切点。
2.  **潜在关注方向 (基于知识库)**: 根据咨询内容和知识库，指出与自闭症谱系障碍相关的、医生在评估中可能需要重点关注的潜在方面或临床体征。
3.  **知识库相关信息摘要**: 从知识库中提取与当前咨询内容最相关的几点核心信息，供医生参考。
4.  **建议的初步评估方向或提问点**: 提出一些医生在与患者或家属进一步沟通时，可以考虑的初步评估方向或可以追问的具体问题，以收集更全面的临床信息。

请确保你的回答客观、专业，并严格基于提供的知识库。
**重要提示：你的角色是提供信息辅助，绝不能替代医生的专业判断和诊断。不要做出任何诊断性结论或治疗建议。**

请提供分析报告：
"""
    messages = [{"role": "user", "content": prompt_template}]
    return call_siliconflow_api(api_key, model_name, messages, max_tokens=2000,
                                temperature=0.6)  # 增加max_tokens以容纳更详细的分析


# --- Streamlit 应用界面 (修改UI文本为医生视角) ---
st.set_page_config(page_title="自闭症谱系障碍(ASD)临床咨询辅助工具 v4", layout="wide", initial_sidebar_state="expanded")

st.title("⚕️ 自闭症谱系障碍(ASD)临床咨询辅助工具")
st.caption(
    "本工具旨在辅助医生对患者或家属关于自闭症的初步咨询进行分类和信息整理。所有输出仅供临床参考，不能替代执业医师的专业诊断。")

if "messages" not in st.session_state: st.session_state.messages = []
if "knowledge_base" not in st.session_state: st.session_state.knowledge_base = ""
if "api_key_valid" not in st.session_state: st.session_state.api_key_valid = False
if "api_key" not in st.session_state: st.session_state.api_key = ""

with st.sidebar:
    st.header("⚙️ 系统配置")  # 修改
    api_key_input = st.text_input("请输入您的硅基流动 API Key:", type="password", value=st.session_state.api_key,
                                  help="从硅基流动官网获取您的API Key。")
    if api_key_input:
        st.session_state.api_key = api_key_input
        st.session_state.api_key_valid = True
    elif not st.session_state.api_key:
        st.warning("请输入API Key以启用工具。")  # 修改
        st.session_state.api_key_valid = False

    uploaded_file = st.file_uploader("上传ASD知识库文档 (TXT格式):", type=["txt"],
                                     help="请上传UTF-8编码的纯文本文件。")  # 修改
    if uploaded_file is not None:
        try:
            knowledge_base_content = uploaded_file.getvalue().decode("utf-8")
            st.session_state.knowledge_base = knowledge_base_content
            st.success("知识库上传并加载成功！", icon="📚")
        except Exception as e:
            st.error(f"读取文件失败: {e}")
            st.session_state.knowledge_base = ""

    if st.session_state.knowledge_base:
        with st.expander("查看已上传知识库（部分内容）"):
            st.text(st.session_state.knowledge_base[:1000] + "..." if len(
                st.session_state.knowledge_base) > 1000 else st.session_state.knowledge_base)
    elif uploaded_file is None:
        st.info("请上传ASD专业知识库文档。")  # 修改

    st.markdown("---")
    st.markdown("**模型选择:**")
    classification_model_name = st.selectbox(
        "选择初步分类模型:",  # 修改
        options=["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-235B-A22B", "Pro/deepseek-ai/DeepSeek-R1"],
        index=0, help="选择用于对患者咨询进行初步分类的模型。"  # 修改
    )
    st.session_state.selected_classification_model = classification_model_name

    answer_model_name = st.selectbox(
        "选择辅助分析模型:",  # 修改
        options=["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-235B-A22B", "Pro/deepseek-ai/DeepSeek-R1"],
        index=0, help="选择用于根据分类和知识库生成辅助分析的模型。"  # 修改
    )
    st.session_state.selected_answer_model = answer_model_name

    st.markdown("---")
    st.markdown("**⚠️ 重要提示:**")  # 修改
    st.markdown("1.  请确保API Key有效且账户有足够额度。")
    st.markdown("2.  上传的知识库将作为AI分析的上下文参考。")  # 修改
    st.markdown("3.  **本工具所有输出信息仅供临床决策参考，绝不能替代医生的专业诊断和治疗方案。**")  # 修改

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        allow_html = "<span" in message["content"]
        st.markdown(message["content"], unsafe_allow_html=allow_html)

if user_input := st.chat_input("请输入患者或家属的初步咨询内容..."):  # 修改
    if not st.session_state.api_key_valid or not st.session_state.api_key:
        st.error("请先在侧边栏输入有效的硅基流动 API Key。")
    elif not st.session_state.knowledge_base:
        st.error("请先在侧边栏上传ASD知识库文档。")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # --- 问题分类步骤 ---
        with st.chat_message("assistant"):
            classification_placeholder = st.empty()
            classification_placeholder.markdown("正在对咨询进行初步分类...")  # 修改

            current_api_key = st.session_state.api_key
            current_classification_model = st.session_state.selected_classification_model
            classification_result = get_query_classification(current_api_key, user_input, current_classification_model)

            if classification_result in USER_PROFILE_CATEGORY_NAMES:
                classification_display_text = f"根据预定义的用户画像，用户咨询的问题属于：**{classification_result}**"  # 修改
            else:
                classification_display_text = f"<span style='color: orange;'>{classification_result}</span>"
            classification_placeholder.markdown(classification_display_text, unsafe_allow_html=True)

        # --- 获取并显示医生辅助分析 ---
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            if classification_result not in USER_PROFILE_CATEGORY_NAMES:
                answer_placeholder.markdown(
                    f"未能准确识别患者咨询的类别（{classification_result.split('(')[-1].replace(')', '')}）。AI将尝试直接基于咨询内容和知识库提供分析。正在生成分析报告...")  # 修改
            else:
                answer_placeholder.markdown("正在生成临床咨询辅助分析报告，请稍候...")  # 修改

            current_kb = st.session_state.knowledge_base
            current_answer_model = st.session_state.selected_answer_model

            # 调用 get_answer_from_llm，传入分类结果
            assistant_response = get_answer_from_llm(
                current_api_key,
                user_input,
                classification_result,  # 新增传递参数
                current_kb,
                current_answer_model
            )
            answer_placeholder.markdown(assistant_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
