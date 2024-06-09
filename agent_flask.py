import logging
import os
from logging.handlers import RotatingFileHandler
from operator import itemgetter
from urllib.parse import quote_plus

from flask import Flask
from flask_cors import CORS
from flask import Response, stream_with_context, request

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.llms import Tongyi
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from config import DASHSCOPE_API_KEY, HOST, NAME, USER, PASSWORD, SYSTEM_TEMPLATE, EXAMPLES, PREFIX


def print_and_pass_through(value, label):
    print(f"{label}: {value}")
    return value


"""
This part initializes the database connection and the LLM model, 
and defines the prompt template and the chain.
"""
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
user = USER
password = PASSWORD
host = HOST
name = NAME
# 这一步是为了防止密码中有特殊字符，如@，导致连接失败
passwd = quote_plus(password)
uri = f"mysql+pymysql://{user}:{passwd}@{host}:3306/{name}"
db = SQLDatabase.from_uri(uri)
llm = Tongyi(model='qwen-max-longcontext', temperature=0)
user_template = """
请你根据以下问题、对应的SQL查询以及SQL查询结果，回答我的问题，如果不是特定的问题，你的回答要尽可能详细一点。
问题: {question}
SQL 查询: {query}
SQL 查询结果: {result}
答案:
"""
system_template = SYSTEM_TEMPLATE

answer_prompt = ChatPromptTemplate.from_messages(
    [("system", system_template),
     ("user", user_template)]
)

example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")
prompt = FewShotPromptTemplate(
    examples=EXAMPLES,
    example_prompt=example_prompt,
    prefix=PREFIX,
    suffix="User input: {input}\nSQL query: ",
    input_variables=["input", "top_k", "table_info"],
)

write_query = create_sql_query_chain(llm=llm, db=db, prompt=prompt)
execute_query = QuerySQLDataBaseTool(db=db)

# 定义链再传给LLM，然后使用StrOutputParser解析输出
answer = answer_prompt | llm | StrOutputParser()
chain = (
        RunnablePassthrough.assign(query=write_query)
        .assign(result=itemgetter("query") | execute_query)
        | (lambda x: print_and_pass_through(x, 'Query'))
        | (lambda x: print_and_pass_through(x, 'Execution Result'))
        | answer
)

"""
flask 相关
"""
app = Flask(__name__)
CORS(app)  # 这将允许所有来源访问所有路由


# hello接口，测试是否正常运行
@app.route('/hello')
def hello():
    return "Hello World!"


# 问答接口，流式返回
@app.route('/dbchat/stream', methods=['GET'])
def chat_stream():
    user_input = request.args.get('user_input', '')

    app.logger.info("user_input: %s", user_input)

    def generate():
        try:
            # 调用模型，得到机器人回复
            res = chain.stream({"question": user_input})
            for i in res:
                # 日志打印
                app.logger.info("res.content: %s", i)
                yield "data: {}\n\n".format(i)
            print("ENDENDEND")
            yield "data: {}\n\n".format("ENDENDEND")
        except Exception as e:
            # 打印更完整的错误信息
            app.logger.exception(e)
            yield "Error, exception"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


def setup_logger():
    handler = RotatingFileHandler('llm4db.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)


if __name__ == '__main__':
    # 使用debug
    setup_logger()
    app.run("0.0.0.0", port=5003)
    # app.run("0.0.0.0", port=5003)
