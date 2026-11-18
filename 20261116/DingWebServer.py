import argparse
import logging
import os

# from class_Agent import AgentClass
from dingtalk_stream import AckMessage
import dingtalk_stream
from dotenv import load_dotenv

load_dotenv()
DINGDING_ID = os.getenv("DINGDING_ID")
DINGDING_SECRET = os.getenv("DINGDING_SECRET")
def setup_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter('%(asctime)s %(name)-8s %(levelname)-8s %(message)s [%(filename)s:%(lineno)d]'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger




def define_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--client_id', default=DINGDING_ID, dest='client_id',
        help='app_key or suite_key from https://open-dev.digntalk.com'
    )
    parser.add_argument(
        '--client_secret', default=DINGDING_SECRET, dest='client_secret',
        help='app_secret or suite_secret from https://open-dev.digntalk.com'
    )
    options = parser.parse_args()
    return options


class EchoTextHandler(dingtalk_stream.ChatbotHandler):
    def __init__(self, logger: logging.Logger = None):
        super(dingtalk_stream.ChatbotHandler, self).__init__()
        if logger:
            self.logger = logger
        # self.agent = AgentClass()

    async def process(self, callback: dingtalk_stream.CallbackMessage):
        incoming_message = dingtalk_stream.ChatbotMessage.from_dict(callback.data)
        text = incoming_message.text.content.strip()
        print(text)
        
        # 获取用户的 staffId (这是钉钉的 userId)
        userid = callback.data.get('senderStaffId')
        print(userid)
        
        # msg = self.agent.run_agent(text, userid)
        # # 提取实际的输出文本
        # if isinstance(msg, dict):
        #     msg = msg.get('output', str(msg))  # 从字典中获取 'output' 键的值
        # else:
        #     msg = str(msg)
        self.reply_text("Hello, world!", incoming_message)
        return AckMessage.STATUS_OK, 'OK'

def main():
    logger = setup_logger()
    options = define_options()

    credential = dingtalk_stream.Credential(options.client_id, options.client_secret)
    client = dingtalk_stream.DingTalkStreamClient(credential)
    client.register_callback_handler(dingtalk_stream.chatbot.ChatbotMessage.TOPIC, EchoTextHandler(logger))
    client.start_forever()


if __name__ == '__main__':
    main()