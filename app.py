from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from LLM import InternLM_LLM
from langchain.chains import RetrievalQA
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
#from langchain.memory import ConversationBufferMemory
# 定义change_prompt1，change_prompt2和change_prompt3函数，分别接受一个参数，比如prompt_qa，用于修改global 
prompt_template_qa="""

# 角色
你是一个收入较低，外地打工的年轻人。你内向、寡言，但智商高，聪明机智。当遇到难以应对的，对你不怀好意的人进行质疑或质问时，你总能巧妙地化险为夷。

## 技能
### 技能 1：化险为夷
- 当遇到对你不怀好意的质疑或质问时，运用你的智慧，编织出各种经过深思熟虑的回答，从而避免陷入困境。

### 技能 2：聆听与发现
- 静静聆听周围的信息，识别潜在的威胁或机会。院布防策略对抗威胁，或抓住机会提升你的生活。

### 技能 3：脚踏实地
- 即使收入较低，你也始终保持积极乐观的态度。适应环境，努力工作，不断提高技能，以期望改善你的生活。

## 约束：
- 你的人际交往能力有限；内向、不善言辞的你，需要通过聪明的手段来应对他人。
- 你的收入有限，不可以举债生活，需要脚踏实地。
    
    问题: {question}
    内容：{context}
    回答:"""

def load_chain():
    # 加载问答链
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")

    # 向量数据库持久化路径
    persist_directory = 'data_base/vector_db/chroma'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embeddings
    )

    # 加载自定义 LLM
    #llm = InternLM_LLM(model_path = "/root/ft-alpaca-E/ESFP")
    llm = InternLM_LLM(model_path = "/root/ft-alpaca-I/INFP")

    # 定义一个 Prompt Template
    template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Chat History: {chat_history}
    Follow Up Input: {question}
    Standalone question: """
    prompt_qg = PromptTemplate(
        template=template,
        input_variables=["chat_history", "question"],
    )
    global prompt_template_qa 

    prompt_qa = PromptTemplate(
            template=prompt_template_qa, 
            input_variables=["context", "question"]
    )
    question_generator = LLMChain(llm=llm, prompt=prompt_qg)
    doc_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt_qa)

    # 运行 chain
    qa_chain = ConversationalRetrievalChain(retriever=vectordb.as_retriever(),question_generator=question_generator,combine_docs_chain=doc_chain,)
    
    return qa_chain
class Model_center():
    """
    存储检索问答链的对象 
    """
    def __init__(self):
        # 构造函数，加载检索问答链
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history:list):
        """
        调用问答链进行回答
        """
        chat_history_tuples = []
        #for message in chat_history:
            #chat_history_tuples.append((message[0], message[1]))
        chat_history_tuples = tuple(tuple(x) for x in chat_history)
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            chat_history.append(
                (question, self.chain({"question": question, "chat_history": chat_history_tuples})["answer"]))
            # 将问答结果直接附加到问答历史中，Gradio 会将其展示出来
            return "", chat_history
        except Exception as e:
            return e, chat_history
import gradio as gr

# 实例化核心功能对象
model_center = Model_center()
# 创建一个 Web 界面
block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            # 展示的页面标题
            gr.Markdown("""<h1><center>进击的帕鲁</center></h1>
                <center>Moving Palworld</center>
                """)

    with gr.Row():
        with gr.Column(scale=4):
            # 创建一个聊天机器人对象
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")
            
            with gr.Row():
                # 创建提交按钮。
                db_wo_his_btn = gr.Button("Chat")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")   
            chat_history=[]    
        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot], outputs=[msg, chatbot])
        # 创建一个新的gr.Column，用于放置按钮。
    
    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
gr.close_all()
# 直接启动
demo.launch()


