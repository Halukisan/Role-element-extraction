from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
import numpy as np
from tqdm import tqdm
import json

# 以下代码为训练数据集的制作，因为平台给与的数据集质量堪忧，并且数据的量也不够，所以对于原有的数据，进行了扩充。
def chatbot(prompt):
    #星火认知大模型Spark3.5 Max的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
    SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.5/chat'
    #星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
    SPARKAI_APP_ID = ''
    SPARKAI_API_SECRET = ''
    SPARKAI_API_KEY = ''
    #星火认知大模型Spark3.5 Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
    SPARKAI_DOMAIN = 'generalv3.5'
    spark = ChatSparkLLM(
        spark_api_url=SPARKAI_URL,
        spark_app_id=SPARKAI_APP_ID,
        spark_api_key=SPARKAI_API_KEY,
        spark_api_secret=SPARKAI_API_SECRET,
        spark_llm_domain=SPARKAI_DOMAIN,
        streaming=False,
    )
    messages = [ChatMessage(
        role="user",
        content=prompt
    )]
    handler = ChunkPrintHandler()
    a = spark.generate([messages], callbacks=[handler])
    return a.generations[0][0].message.content

content = ''
prompt = f'''
你是一个数据分析大师，你需要从群聊对话中进行分析，里面对话的角色中大部分是客服角色，你需要从中区分出有需求的客户，并得到以下四类数据。

****群聊对话****
{content}

****分析数据****
客户基本信息：需要从中区分出客户角色，并得到客户基本信息，其中包括姓名、手机号码、邮箱、地区、详细地址、性别、年龄和生日
客户意向与预算信息： 客户意向与预算信息包括咨询类型、意向产品、购买异议点、预算是否充足、总体预算金额以及预算明细
客户购买准备情况：户购买准备情况包括竞品信息、客户是否有意向、客户是否有卡点以及客户购买阶段
跟进计划信息： 跟进计划信息包括参与人、时间点和具体事项，这些信息用于指导销售团队在未来的跟进工作中与客户互动

****注意****
1.只输出客户基本信息、客户意向与预算信息、客户购买准备情况、跟进计划信息对应的信息，不要输出无关内容
2.不要输出分析内容
3.输出内容格式为md格式
'''

jsonl_data = {"instruction":"假设你是一个智能交互助手，基于用户的输入文本，解析其中语义，抽取关键信息，以json格式生成结构化的语义内容。","input":"请调小空气净化器的湿度到1","output":"{\"intent\":\"CONTROL\",\"slots\":[{\"name\":\"device\",\"normValue\":\"airCleaner\",\"value\":\"空气净化器\"},{\"name\":\"insType\",\"normValue\":\"set\",\"value\":\"调小\"},{\"name\":\"attr\",\"normValue\":\"humidity\",\"value\":\"湿度\"},{\"name\":\"attrValue\",\"normValue\":\"1\",\"value\":\"1\"}],\"sample\":\"请调小空气净化器的湿度到1\"}"}

print(jsonl_data)
print(jsonl_data["instruction"])
print(jsonl_data["input"])
print(jsonl_data["output"])

# 打开并读取JSON文件
with open('train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 训练集制作

# 打开一个文件用于写入，如果文件已存在则会被覆盖
with open('traindata.jsonl', 'w', encoding='utf-8') as file:
    # 训练集行数(130)不符合要求，范围：1500~90000000
    # 遍历数据列表，并将每一行写入文件
    # 这里为了满足微调需求我们重复12次数据集 130*12=1560
    
    for line_data in tqdm(data):
        line_input = line_data["chat_text"] 
        line_output = line_data["infos"]
        content = line_input
        
        prompt = f'''
                你是一个数据分析大师，你需要从群聊对话中进行分析，里面对话的角色中大部分是客服角色，你需要从中区分出有需求的客户，并得到以下四类数据。

                ****群聊对话****
                {content}

                ****分析数据****
                客户基本信息：需要从中区分出客户角色，并得到客户基本信息，其中包括姓名、手机号码、邮箱、地区、详细地址、性别、年龄和生日
                客户意向与预算信息： 客户意向与预算信息包括咨询类型、意向产品、购买异议点、预算是否充足、总体预算金额以及预算明细
                客户购买准备情况：户购买准备情况包括竞品信息、客户是否有意向、客户是否有卡点以及客户购买阶段
                跟进计划信息： 跟进计划信息包括参与人、时间点和具体事项，这些信息用于指导销售团队在未来的跟进工作中与客户互动

                ****注意****
                1.只输出客户基本信息、客户意向与预算信息、客户购买准备情况、跟进计划信息对应的信息，不要输出无关内容
                2.不要输出分析内容
                3.输出内容格式为md格式
                '''
        res = chatbot(prompt=prompt)
        # print(res)
        line_write = {
            "instruction":jsonl_data["instruction"],
            "input":json.dumps(res, ensure_ascii=False),
            "output":json.dumps(line_output, ensure_ascii=False)
        }
        # 因为数据共有130行，为了能满足训练需要的1500条及以上，我们将正常训练数据扩充12倍。
        for time in range(12):
            file.write(json.dumps(line_write, ensure_ascii=False) + '\n')  # '\n' 用于在每行末尾添加换行符





# 验证集制作（提交版本）
# input,target

import json

# 打开并读取JSON文件
with open('test_data.json', 'r', encoding='utf-8') as file:
    data_test = json.load(file)



import csv

# 打开一个文件用于写入CSV数据
with open('test.csv', 'w', newline='', encoding='utf-8') as csvfile:
    # 创建一个csv writer对象
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["input","target"])
    # 遍历数据列表，并将每一行写入CSV文件
    for line_data in tqdm(data_test):
        content = line_data["chat_text"]
        prompt = f'''
                你是一个数据分析大师，你需要从群聊对话中进行分析，里面对话的角色中大部分是客服角色，你需要从中区分出有需求的客户，并得到以下四类数据。

                ****群聊对话****
                {content}

                ****分析数据****
                客户基本信息：需要从中区分出客户角色，并得到客户基本信息，其中包括姓名、手机号码、邮箱、地区、详细地址、性别、年龄和生日
                客户意向与预算信息： 客户意向与预算信息包括咨询类型、意向产品、购买异议点、预算是否充足、总体预算金额以及预算明细
                客户购买准备情况：户购买准备情况包括竞品信息、客户是否有意向、客户是否有卡点以及客户购买阶段
                跟进计划信息： 跟进计划信息包括参与人、时间点和具体事项，这些信息用于指导销售团队在未来的跟进工作中与客户互动

                ****注意****
                1.只输出客户基本信息、客户意向与预算信息、客户购买准备情况、跟进计划信息对应的信息，不要输出无关内容
                2.不要输出分析内容
                3.输出内容格式为md格式
                '''
        res = chatbot(prompt=prompt)
        
        # print(line_data["chat_text"])
        ## 文件内容校验失败: test.jsonl(不含表头起算)第1行的内容不符合规则，限制每组input和target字符数量总和上限为8000，当前行字符数量：10721
        line_list = [res, "-"]   
        csvwriter.writerow(line_list)
        # break



