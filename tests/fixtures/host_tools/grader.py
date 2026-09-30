"""Guest package entry point: no grader behavior is built into shellsim."""

import base64

from host_tools import ToolClient


class Container:
    def __init__(self, client):
        self.client = client

    def read_file(self, path):
        result = self.client.call("workspace.read_file", {"path": path})
        return base64.b64decode(result["data_base64"])


class ChatContext:
    def __init__(self, conversation, container):
        self.conversation = conversation
        self.container = container


class Grader:
    def grade(self, ctx):
        answer = ctx.container.read_file("/work/answer.txt").decode("utf-8").strip()
        return 1.0 if ctx.conversation == ["What is 6 * 7?"] and answer == "42" else 0.0


client = ToolClient()
conversation = client.call("conversation.list", {})["messages"]
context = ChatContext(conversation, Container(client))
score = Grader().grade(context)
client.call("grade.submit", {"score": score})
print(score)
