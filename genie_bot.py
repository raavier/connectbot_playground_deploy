import json
from typing import Any, Callable, Generator
from uuid import uuid4

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from openai import OpenAI
from pydantic import BaseModel

# Configurações
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
GENIE_SPACE_ID = "SEU_SPACE_ID_AQUI"  # ← Substituir pelo seu Space ID

SYSTEM_PROMPT = """Você é um assistente que responde perguntas sobre dados CRM usando o Genie Space.
Para qualquer pergunta sobre dados, use a ferramenta query_genie para obter a resposta."""

class ToolInfo(BaseModel):
    name: str
    spec: dict
    exec_fn: Callable

def create_tool_info(tool_spec, exec_fn_param=None):
    tool_spec["function"].pop("strict", None)
    tool_name = tool_spec["function"]["name"]
    udf_name = tool_name.replace("__", ".")

    def exec_fn(**kwargs):
        from unitycatalog.ai.core.base import get_uc_function_client
        uc_function_client = get_uc_function_client()
        function_result = uc_function_client.execute_function(udf_name, kwargs)
        if function_result.error is not None:
            return function_result.error
        return function_result.value

    return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)

# UC Function: query_genie
UC_TOOL_NAMES = ["hs_franquia.gold_connect_bot.query_genie"]
uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
TOOL_INFOS = [create_tool_info(tool_spec) for tool_spec in uc_toolkit.tools]

class GenieBot(ResponsesAgent):
    def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client = self.workspace_client.serving_endpoints.get_open_ai_client()
        self._tools_dict = {tool.name: tool for tool in tools}

    def get_tool_specs(self) -> list[dict]:
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        return self._tools_dict[tool_name].exec_fn(**args)

    def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        for chunk in self.model_serving_client.chat.completions.create(
            model=self.llm_endpoint,
            messages=to_chat_completions_input(messages),
            tools=self.get_tool_specs(),
            stream=True,
        ):
            chunk_dict = chunk.to_dict()
            if len(chunk_dict.get("choices", [])) > 0:
                yield chunk_dict

    def handle_tool_call(self, tool_call: dict[str, Any], messages: list[dict[str, Any]]) -> ResponsesAgentStreamEvent:
        args = json.loads(tool_call["arguments"])
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))

        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)

    def call_and_run_tools(self, messages: list[dict[str, Any]], max_iter: int = 10) -> Generator[ResponsesAgentStreamEvent, None, None]:
        for _ in range(max_iter):
            last_msg = messages[-1]
            if last_msg.get("role", None) == "assistant":
                return
            elif last_msg.get("type", None) == "function_call":
                yield self.handle_tool_call(last_msg, messages)
            else:
                yield from output_to_responses_items_stream(
                    chunks=self.call_llm(messages), aggregator=messages
                )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item("Max iterations reached.", str(uuid4())),
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        messages = to_chat_completions_input([i.model_dump() for i in request.input])
        if SYSTEM_PROMPT:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        yield from self.call_and_run_tools(messages=messages)

# Inicializar agent
mlflow.openai.autolog()
AGENT = GenieBot(llm_endpoint=LLM_ENDPOINT_NAME, tools=TOOL_INFOS)
mlflow.models.set_model(AGENT)
