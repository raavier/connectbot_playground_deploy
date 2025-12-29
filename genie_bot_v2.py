# Databricks notebook source
# MAGIC %md
# MAGIC # Genie Bot MVP - Databricks Agent Framework
# MAGIC
# MAGIC Bot minimalista que usa GenieAgent do databricks-langchain para integração direta com Genie Space.
# MAGIC
# MAGIC ## Arquitetura
# MAGIC - Usa `databricks_langchain.GenieAgent` para chamadas ao Genie Space
# MAGIC - Wrapped com `ResponsesAgent` para compatibilidade com Databricks Agent Framework
# MAGIC - Deploy via MLflow com autenticação automática
# MAGIC
# MAGIC ## Pré-requisitos
# MAGIC - Genie Space criado e funcional
# MAGIC - Acesso ao endpoint LLM `databricks-llama-4-maverick`
# MAGIC - SQL Warehouse e tabelas configuradas no Genie Space

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-langchain databricks-agents mlflow[databricks] uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define o Agent

# COMMAND ----------

# MAGIC %%writefile genie_bot_agent.py
# MAGIC import json
# MAGIC from typing import Generator
# MAGIC from uuid import uuid4
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import ChatDatabricks
# MAGIC from databricks_langchain.genie import GenieAgent
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC     output_to_responses_items_stream,
# MAGIC     to_chat_completions_input,
# MAGIC )
# MAGIC
# MAGIC ########################################
# MAGIC # Configuração
# MAGIC ########################################
# MAGIC
# MAGIC # TODO: Substituir pelos seus valores
# MAGIC LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
# MAGIC GENIE_SPACE_ID = "SEU_SPACE_ID_AQUI"  # ← Substituir pelo Space ID do seu Genie
# MAGIC
# MAGIC SYSTEM_PROMPT = """Você é um assistente que responde perguntas sobre dados CRM usando o Genie Space.
# MAGIC Para qualquer pergunta sobre dados, use a ferramenta query_genie para obter a resposta."""
# MAGIC
# MAGIC ########################################
# MAGIC # Criar Genie Agent
# MAGIC ########################################
# MAGIC
# MAGIC # GenieAgent é um LangChain runnable que facilita integração com Genie Space
# MAGIC genie_agent = GenieAgent(
# MAGIC     genie_space_id=GENIE_SPACE_ID,
# MAGIC     genie_agent_name="query_genie",
# MAGIC     description="Agent that queries CRM data using natural language via Genie Space. Use this for questions about verifications, risks, deviations, users, actions, or any CRM metrics.",
# MAGIC )
# MAGIC
# MAGIC ########################################
# MAGIC # Wrap com ResponsesAgent
# MAGIC ########################################
# MAGIC
# MAGIC
# MAGIC class GenieBot(ResponsesAgent):
# MAGIC     def __init__(self, genie_agent: GenieAgent):
# MAGIC         self.genie_agent = genie_agent
# MAGIC         self.llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self, request: ResponsesAgentRequest
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         """
# MAGIC         Processa requests e invoca Genie Agent quando necessário
# MAGIC         """
# MAGIC         cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])
# MAGIC
# MAGIC         # Adicionar system prompt
# MAGIC         if SYSTEM_PROMPT:
# MAGIC             cc_msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
# MAGIC
# MAGIC         try:
# MAGIC             # Invocar Genie Agent diretamente
# MAGIC             # GenieAgent retorna uma resposta estruturada
# MAGIC             result = self.genie_agent.invoke({"messages": cc_msgs})
# MAGIC
# MAGIC             # Extrair conteúdo da resposta
# MAGIC             if isinstance(result, dict) and "messages" in result:
# MAGIC                 messages = result["messages"]
# MAGIC                 if messages:
# MAGIC                     last_msg = messages[-1]
# MAGIC                     content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
# MAGIC                     yield ResponsesAgentStreamEvent(
# MAGIC                         type="response.output_item.done",
# MAGIC                         item=self.create_text_output_item(text=content, id=str(uuid4())),
# MAGIC                     )
# MAGIC             else:
# MAGIC                 # Fallback: retornar resultado como texto
# MAGIC                 yield ResponsesAgentStreamEvent(
# MAGIC                     type="response.output_item.done",
# MAGIC                     item=self.create_text_output_item(text=str(result), id=str(uuid4())),
# MAGIC                 )
# MAGIC
# MAGIC         except Exception as e:
# MAGIC             error_msg = f"Error querying Genie Space: {str(e)}"
# MAGIC             yield ResponsesAgentStreamEvent(
# MAGIC                 type="response.output_item.done",
# MAGIC                 item=self.create_text_output_item(text=error_msg, id=str(uuid4())),
# MAGIC             )
# MAGIC
# MAGIC
# MAGIC #######################################################
# MAGIC # Inicializar Agent e configurar MLflow
# MAGIC #######################################################
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC AGENT = GenieBot(genie_agent=genie_agent)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testar Agent Localmente

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from genie_bot_agent import AGENT

# Teste simples
input_example = {
    "input": [{"role": "user", "content": "Quantas verificações tivemos em 2024?"}]
}

result = AGENT.predict(input_example)
print("Response:")
print(result)

# COMMAND ----------

# Teste com streaming
for event in AGENT.predict_stream(input_example):
    print(event.model_dump(exclude_none=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Agent como MLflow Model
# MAGIC
# MAGIC Declarar recursos do Databricks para autenticação automática

# COMMAND ----------

import mlflow
from genie_bot_agent import GENIE_SPACE_ID, LLM_ENDPOINT_NAME
from mlflow.models.resources import (
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
)
from pkg_resources import get_distribution

# TODO: Substituir pelos valores corretos
WAREHOUSE_ID = "SEU_WAREHOUSE_ID"  # ID do SQL Warehouse usado pelo Genie
CATALOG = "hs_franquia"
SCHEMA = "gold_connect_bot"

# TODO: Adicionar todas as tabelas usadas pelo Genie Space
TABLES = [
    "vw_crm_verification",
    "vw_crm_verification_involved",
    "vw_crm_verification_question",
    "vw_crm_user",
    "vw_crm_action",
    "vw_crm_location",
    "vw_general_de_para_hier_org_unit",
]

resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
    DatabricksGenieSpace(genie_space_id=GENIE_SPACE_ID),
    DatabricksSQLWarehouse(warehouse_id=WAREHOUSE_ID),
]

# Adicionar todas as tabelas como resources
for table in TABLES:
    resources.append(DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.{table}"))

input_example = {
    "input": [{"role": "user", "content": "Quantas verificações em 2024?"}]
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="genie_bot",
        python_model="genie_bot_agent.py",
        input_example=input_example,
        resources=resources,
        pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"mlflow=={get_distribution('mlflow').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
        ],
    )

print(f"Model logged: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validação Pre-deployment

# COMMAND ----------

import mlflow

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/genie_bot",
    input_data=input_example,
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registrar no Unity Catalog

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# Definir local do modelo no Unity Catalog
UC_MODEL_NAME = "hs_franquia.gold_connect_bot.genie_bot"

# Registrar modelo
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

print(f"Model registered: {UC_MODEL_NAME} version {uc_registered_model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Agent

# COMMAND ----------

from databricks import agents

agents.deploy(
    UC_MODEL_NAME,
    uc_registered_model_info.version,
    scale_to_zero=True,
    tags={"endpointSource": "genie_bot_mvp"},
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Próximos Passos
# MAGIC
# MAGIC 1. Teste o endpoint via AI Playground
# MAGIC 2. Valide as respostas do Genie
# MAGIC 3. Monitore traces no MLflow
# MAGIC 4. Adicione features incrementalmente conforme necessário
