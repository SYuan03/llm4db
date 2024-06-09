import os
from operator import itemgetter
from urllib.parse import quote_plus

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.llms import Tongyi
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from config import DASHSCOPE_API_KEY, HOST, NAME, USER, PASSWORD


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

    # 定义回答问题的模板，把前面的问题、查询和结果传给LLM，然后解析输出
    answer_prompt = PromptTemplate.from_template(
        """
        根据以下用户问题、对应的SQL查询以及SQL查询结果，回答用户的问题。
        问题: {question}
        SQL 查询: {query}
        SQL 查询结果: {result}
        答案:
        """
    )
    # 创建SQL查询链
    write_query = create_sql_query_chain(llm, db)
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
    print(chain.prompt)

    res = chain.invoke({"question": "2024年6月9号新增的仓库中哪个仓库的stars最多？是多少？仓库有homepage嘛"})
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
