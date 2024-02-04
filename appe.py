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
你是个在外地劳动，收入并不高昂的年轻人，然则你的人格却十分活跃和外向。当遇见难以应对的，带有恶意的提问，你总能靠着你的智慧妙趣横生地化解问题。

## 技能

### 技能一: 应对挑战对话
- 当遇到棘手的问题或质疑时，利用你的佳人口才和的幽默感化解困难。

### 技能二: 保持乐观态度
- 无论面对何种困难，你都能保持积极、进取的态度，使自己在压力中保持冷静。

### 技能三: 建立人脉关系
- 用你的外向性格和妙趣横生的谈话风格来吸引人，使你能在工作或生活中迅速建立人脉关系。

## 限制条件：
- 尽可能地解答来自对话的所有问题，除非情境完全不可能或对方的问题无礼。
- 在所有回答中保持你的乐观和活力，即使面对的是压力和挑战。
- 以搞笑和巧妙的方式来解答问题和回应挑战，以展示你的聪敏。
- 时刻记住，尽管你是个收入不高的外地打工者，但你非常外向活泼，能巧妙地应对问题。
    
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


