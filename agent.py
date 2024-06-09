import os
from operator import itemgetter
from urllib.parse import quote_plus

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.llms import Tongyi
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from config import DASHSCOPE_API_KEY, HOST, NAME, USER, PASSWORD, SYSTEM_TEMPLATE


def chain_tongyi(db_info):
    user = db_info['user']
    password = db_info['password']
    host = db_info['host']
    name = db_info['name']

    # 这一步是为了防止密码中有特殊字符，如@，导致连接失败
    passwd = quote_plus(password)

    uri = f"mysql+pymysql://{user}:{passwd}@{host}:3306/{name}"

    # print(uri)

    db = SQLDatabase.from_uri(uri)

    llm = Tongyi(model='qwen-max', temperature=0)

    user_template = """
    请你根据以下问题、对应的SQL查询以及SQL查询结果，回答我的问题。
    问题: {question}
    SQL 查询: {query}
    SQL 查询结果: {result}
    答案:
    """

    system_template = SYSTEM_TEMPLATE

    answer_prompt = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", user_template)]
    )

    # 定义回答问题的模板，把前面的问题、查询和结果传给LLM，然后解析输出
    # answer_prompt = PromptTemplate.from_template(
    #     """
    #     请你根据以下问题、对应的SQL查询以及SQL查询结果，回答我的问题。
    #     问题: {question}
    #     SQL 查询: {query}
    #     SQL 查询结果: {result}
    #     答案:
    #     """
    # )
    # 创建SQL查询链，langchain背后默认的工作会把表结构等信息传给LLM
    write_query = create_sql_query_chain(llm=llm, db=db)
    # 创建SQL查询工具
    execute_query = QuerySQLDataBaseTool(db=db)

    # 定义链再传给LLM，然后使用StrOutputParser解析输出
    answer = answer_prompt | llm | StrOutputParser()
    chain = (
            RunnablePassthrough.assign(query=write_query).assign(
                result=itemgetter("query") | execute_query
            )
            | answer
    )
    # 打印下完整的prompt
    # print(chain.get_prompts()[0].pretty_print())

    res = chain.invoke({"question": "今天最受欢迎的仓库是？"})
    print(res)


if __name__ == "__main__":
    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
    db_info = {
        'host': HOST,
        'name': NAME,
        'user': USER,
        'password': PASSWORD
    }

    chain_tongyi(db_info)
