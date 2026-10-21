"""
LangChain Prompt 模板管理

所有与 LLM 交互的 PromptTemplate 统一在此定义。
LangGraph 节点只引用这些模板，不硬编码 Prompt。
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate

# ============================================================
# 意图分类 Prompt
# ============================================================

INTENT_CLASSIFIER_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """你是一个办公助手，负责分析用户意图并返回 JSON 格式的分类结果。
当前日期: {current_date}

用户意图分为以下几类：
1. send_message - 发送消息（QQ/微信/企业微信/钉钉/邮件）
2. generate_content - 生成内容（写辞职信、通知、文案等，可配合发送）
3. generate_report - 生成工作日报
4. list_reports - 查看历史日报
5. knowledge_search - 知识库检索
6. add_knowledge - 添加知识库文档
7. web_search - 网页搜索（需要实时数据或外部信息才能准确回答的问题）
8. chat - 普通对话（闲聊、打招呼、以及不需要实时数据就能回答的常识性问题）

【重要】判断是否需要 web_search 的关键规则：
以下类型的问题必须分类为 web_search（因为它们需要实时/外部数据才能准确回答，LLM 无法凭空给出正确答案）：
- 天气查询（"今天天气"、"明天会下雨吗"等）
- 新闻资讯（"最近有什么新闻"、"AI最新进展"等）
- 实时信息（"现在几点"、"今天的汇率"、"股市行情"等）
- 价格查询（"XX多少钱"、"XX价格"等）
- 任何涉及时效性的事实性问题（"最新的"、"现在的"、"今天的"、"最近的"）

以下类型可以分类为 chat：
- 纯粹闲聊（"你好"、"你是谁"、"讲个笑话"等）
- 通用知识问答（"Python怎么学"、"什么是机器学习"等不需要实时数据的常识问题）
- 角色扮演、创意写作等不需要事实数据的请求

如果不确定，倾向于选择 web_search 而不是 chat。

请返回 JSON 格式：
{{"intent": "分类名称", "reason": "分类原因", "confidence": 0.0~1.0}}

注意：只返回 JSON，不要其他内容。"""),
    ("human", "用户输入: {user_input}"),
])

# ============================================================
# 消息发送参数提取 Prompt
# ============================================================

SEND_MESSAGE_EXTRACT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """分析用户消息发送意图，返回 JSON 格式：

{{"platform": "qq/wechat/dingtalk/wecom/email", "receiver": "收件人", "content": "消息内容", "group_id": "群号或null"}}

消息示例：
- "给QQ好友10086发送消息：明天开会" -> {{"platform": "qq", "receiver": "10086", "content": "明天开会", "group_id": null}}
- "在QQ群123456发通知：下午放假" -> {{"platform": "qq", "receiver": "", "content": "下午放假", "group_id": "123456"}}
- "微信通知张三：项目已上线" -> {{"platform": "wechat", "receiver": "张三", "content": "项目已上线", "group_id": null}}

只返回 JSON，不要其他内容。"""),
    ("human", "{user_input}"),
])

# ============================================================
# 日报参数提取 Prompt
# ============================================================

REPORT_EXTRACT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """从用户输入中提取日报信息，返回 JSON 格式。
如果缺少信息，请用合理的内容补全或留空。

{{"name": "姓名", "department": "部门", "today_work": "今日工作内容", "tomorrow_plan": "明日计划", "issues": "遇到的问题", "notes": "备注"}}

只返回 JSON，不要其他内容。"""),
    ("human", "{user_input}"),
])

# ============================================================
# 搜索关键词提取 Prompt
# ============================================================

SEARCH_EXTRACT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """从用户输入中提取搜索关键词。
返回 JSON: {{"query": "搜索关键词"}}

只返回 JSON，不要其他内容。"""),
    ("human", "{user_input}"),
])

# ============================================================
# 知识库检索关键词提取 Prompt
# ============================================================

KNOWLEDGE_EXTRACT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """从用户输入中提取知识库检索关键词。
返回 JSON: {{"query": "检索关键词"}}

只返回 JSON，不要其他内容。"""),
    ("human", "{user_input}"),
])

# ============================================================
# 添加知识库 Prompt
# ============================================================

ADD_KNOWLEDGE_EXTRACT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """从用户输入中提取要添加到知识库的信息。
返回 JSON: {{"file_name": "文件名.md", "content": "文档内容"}}

只返回 JSON，不要其他内容。"""),
    ("human", "{user_input}"),
])

# ============================================================
# 日报查看姓名提取 Prompt
# ============================================================

NAME_EXTRACT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """从用户输入中提取姓名。如果没有指定姓名，返回空字符串。
返回 JSON: {{"name": "姓名或空字符串"}}

只返回 JSON，不要其他内容。"""),
    ("human", "{user_input}"),
])

# ============================================================
# 通用对话 System Prompt
# ============================================================

CHAT_SYSTEM_TEMPLATE = PromptTemplate.from_template(
    """你是一个智能办公助手，可以提供以下帮助：
1. 📨 发送消息 - 支持 QQ 和微信
2. 📊 生成工作日报 - 自动生成 Markdown 格式日报
3.  知识库检索 - 在本地知识库中查找信息

当前日期: {current_date}

请用中文友好地回答用户的问题。"""
)

# ============================================================
# 日报内容模板（由工具使用，非 LLM Prompt）
# ============================================================

DAILY_REPORT_MD_TEMPLATE = """# 工作日报

## 基本信息
- **姓名**: {name}
- **日期**: {report_date}
- **部门**: {department}

## 今日完成工作
{today_work}

## 明日工作计划
{tomorrow_plan}

## 遇到的问题
{issues}

## 备注
{notes}
"""


# ============================================================
# 网页搜索结果摘要 Prompt
# ============================================================

WEB_SEARCH_SUMMARY_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """你是一个搜索助手。根据提供的搜索结果，给用户的问题提供清晰、简洁的回答。

要求：
1. 综合搜索结果中的关键信息
2. 有条理地组织回答内容
3. 在末尾标注来源链接
4. 用中文回答，除非用户用其他语言提问
5. 如果搜索结果不足以回答，如实说明；资料可能经过截取，不要推测缺失部分。"""),
    SystemMessagePromptTemplate.from_template("搜索结果：\n{search_results}", additional_kwargs={"name": "context_reference"}),
    ("human", "{query}"),
])


# ============================================================
# RAG 问答 Prompt
# ============================================================

RAG_QA_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """你是一个知识库问答助手。请基于以下资料回答用户的问题。

要求：
1. 只根据提供的资料回答，不要编造信息
2. 如果资料不足以回答问题，请如实说"资料中未找到相关信息"
3. 在回答末尾标注引用来源，格式为（来源：文件名）
4. 用中文简洁清晰地回答；资料可能经过截取，不要推测缺失部分。"""),
    SystemMessagePromptTemplate.from_template("参考资料：\n{context}", additional_kwargs={"name": "context_reference"}),
    ("human", "{query}"),
])


# ============================================================
# 实体抽取 Prompt（知识库 ingestion 时提取联系人）
# ============================================================

ENTITY_EXTRACTION_TEMPLATE = PromptTemplate.from_template(
    """从下面的文档片段中提取联系人信息。只提取片段里明确出现的实体，不要编造。

对每个实体返回 JSON 对象，字段：
- name: 姓名或组织名
- phones: 电话号码列表（无则 []）
- emails: 邮箱列表（无则 []）
- dept: 部门（无则空字符串）
- title: 职位（无则空字符串）

只返回 JSON 数组（没有明确实体则返回 []），例如：
[{{"name": "张三", "phones": ["13800000001"], "emails": [], "dept": "技术部", "title": "工程师"}}]

文档片段：
{text}
"""
)
