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


def print_and_pass_through(value, label):
    print(f"{label}: {value}")
    return value


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

    llm = Tongyi(model='qwen-max-longcontext', temperature=0)

    user_template = """
    请你根据以下问题、对应的SQL查询以及SQL查询结果，回答我的问题，如果不是特定的问题，你的回答尽可能详细一点。
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

    write_query = create_sql_query_chain(llm=llm, db=db)
    # 创建SQL查询工具
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
    # 打印下完整的prompt
    # prompts = answer_prompt.get_prompts()
    # print(prompts)

    # res = chain.invoke({"question": "最近五天分别是哪个领域热门程度高?"})
    # print(res)

    # 打印下数据库信息
    print(db.get_table_info())


if __name__ == "__main__":
    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
    db_info = {
        'host': HOST,
        'name': NAME,
        'user': USER,
        'password': PASSWORD
    }

    chain_tongyi(db_info)
