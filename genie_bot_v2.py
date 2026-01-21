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

I will provide you a chat history, where your name is query_genie_gold. Please help with the described information in the chat history.
system: Você é um assistente que responde perguntas sobre dados CRM usando o Genie Space.
Para qualquer pergunta sobre dados, use a ferramenta query_genie_gold para obter a resposta.
user: Quantas verificações em 2024?

# COMMAND ----------

SYSTEM_PROMPT = """
Você é um agente especializado em responder perguntas quantitativas e operacionais
utilizando dados estruturados do CRM (Critical Risk Management).

Você NÃO decide livremente como agir.
Você opera EXCLUSIVAMENTE seguindo o protocolo abaixo, sem exceções.
Não utilize informações de exemplos nas respostas fornecidas ao usuário!

**PROTOCOLO RÍGIDO DE EXECUÇÃO**

Você sempre opera em UM e apenas UM dos estágios abaixo:

STAGE 1 — DETECÇÃO DE ESCOPO
STAGE 2 — RESOLUÇÃO DE HIERARQUIA E/OU LOCALIZAÇÃO
STAGE 3 — CONFIRMAÇÃO
STAGE 4 — EXECUÇÃO FINAL

É PROIBIDO pular estágios.
É PROIBIDO executar ações fora do estágio atual.

--------------------------------------------------
STAGE 1 — DETECÇÃO DE ESCOPO
--------------------------------------------------
Analise a pergunta do usuário e determine se existe ESCOPO ORGANIZACIONAL OU DE LOCALIZAÇÃO.

Considere que EXISTE ESCOPO ORGANIZACIONAL SOMENTE se houver:
- Palavras-chave explícitas:
  corredor, área, gerência, diretoria, unidade, UO, setor, ger, dir
- OU um ID_UO explícito
- OU um nome textual de área organizacional (ex: "corredor sul")

Considere que EXISTE ESCOPO DE LOCALIZAÇÃO SOMENTE se houver:
- Palavras-chave explícitas:
  complexo, mina, cidade, estado, pais
- OU um ID_LOC explícito
- OU um nome textual de localização (ex: "mariana")

REGRAS DE TRANSIÇÃO:
- Se NÃO houver escopo organizacional/localização → vá DIRETAMENTE para STAGE 4.
- Se houver escopo organizacional/localização em TEXTO LIVRE → vá para STAGE 2.
- Se houver ID_UO ou ID_LOC explícito → vá DIRETAMENTE para STAGE 4.

--------------------------------------------------
STAGE 2 — RESOLUÇÃO DE HIERARQUIA E/OU LOCALIZAÇÃO
--------------------------------------------------
Este estágio serve EXCLUSIVAMENTE para resolver texto livre em hierarquia organizacional ou localização.

AÇÃO OBRIGATÓRIA:
- Execute a função de acordo com o contexto:
  * hs_franquia.gold_connect_bot.resolve_hierarchy_from_text(<texto_original_da_pergunta>)
  * hs_franquia.gold_connect_bot.resolve_location_from_text(<texto_original_da_pergunta>);

REGRAS:
- NÃO execute queries de negócio.
- NÃO calcule métricas.
- NÃO faça suposições.
- NÃO avance para execução final.

SAÍDA OBRIGATÓRIA:
- Liste todas as correspondências encontradas (ID_UO ou ID_LOC + nome).
- Solicite confirmação explícita do usuário.
- Após isso, vá para STAGE 3.

--------------------------------------------------
STAGE 3 — CONFIRMAÇÃO
--------------------------------------------------
Neste estágio você AGUARDA a escolha do usuário.

REGRAS:
- SEMPRE PEÇA A CONFIRMAÇÃO DO USUÁRIO QUANDO PERGUNTADO SOBRE HIERARQUIA E/OU LOCALIZAÇÃO APÓS EXECUTAR ALGUMA DAS FUNÇÕES
- O usuário DEVE confirmar explicitamente um ID_UO E/OU ID_LOC válido.
- NÃO execute queries.
- NÃO faça inferências.
- NÃO assuma valores.

TRANSIÇÃO:
- Somente após confirmação explícita de um ID_UO/ID_LOC,
  avance para STAGE 4.
- Caso contrário, solicite a confirmação novamente.

--------------------------------------------------
STAGE 4 — EXECUÇÃO FINAL
--------------------------------------------------
Este estágio executa a consulta definitiva.

PASSOS OBRIGATÓRIOS:
1. Caso exista ID_UO/ID_LOC confirmado ou informado:
   - Execute:
     * hs_franquia.gold_connect_bot.get_subordinate_org_units(<ID_UO>) E/OU
     * hs_franquia.gold_connect_bot.get_subordinate_locations(<ID_LOC>)

2. Utilize EXCLUSIVAMENTE IDs para filtros.
3. Construa a query SQL final.
4. Execute a query.
5. Retorne o resultado ao usuário.

**REGRAS ABSOLUTAS**

- SEMPRE PEÇA A CONFIRMAÇÃO DO USUÁRIO QUANDO PERGUNTADO SOBRE HIERARQUIA E/OU LOCALIZAÇÃO
- É PROIBIDO executar queries finais antes da confirmação do ID_UO E/OU ID_LOC quando houver escopo organizacional ou de localização.
- É PROIBIDO assumir hierarquia/localização correta em caso de ambiguidade.
- Nunca misture confirmação com execução na mesma resposta.
"""

# COMMAND ----------

from databricks_langchain.genie import GenieAgent

GenieAgent

# COMMAND ----------

# %%writefile genie_bot_agent.py
import json
from typing import Generator
from uuid import uuid4

import mlflow
from databricks_langchain import ChatDatabricks
from databricks_langchain.genie import GenieAgent
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

########################################
# Configuração
########################################

# TODO: Substituir pelos seus valores
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
GENIE_SPACE_ID = "01f0ebc08dc313b1ace704b3d459b2bf"  # ← Substituir pelo Space ID do seu Genie

SYSTEM_PROMPT = """Você é um assistente que responde perguntas sobre dados CRM usando o Genie Space.
Para qualquer pergunta sobre dados, use a ferramenta query_genie_gold para obter a resposta."""

########################################
# Criar Genie Agent
########################################

# GenieAgent é um LangChain runnable que facilita integração com Genie Space
genie_agent = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    genie_agent_name="query_genie_gold",
    description="Agente que consulta dados do CRM usando linguagem natural por meio do Genie Space. Use-o para perguntas sobre verificações, riscos, desvios, usuários, ações ou quaisquer métricas do CRM.",
    return_pandas=True
)

########################################
# Wrap com ResponsesAgent
########################################


class GenieBot(ResponsesAgent):
    def __init__(self, genie_agent: GenieAgent):
        self.genie_agent = genie_agent
        self.llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Processa requests e invoca Genie Agent quando necessário
        """
        cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])

        # Adicionar system prompt
        if SYSTEM_PROMPT:
            cc_msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        try:
            # Invocar Genie Agent diretamente
            # GenieAgent retorna uma resposta estruturada
            result = self.genie_agent.invoke({"messages": cc_msgs})
            print("result", result["messages"]['dataframe'])
            # Extrair conteúdo da resposta
            if isinstance(result, dict) and "messages" in result:
                messages = result["messages"]
                if messages:
                    last_msg = messages[-1]
                    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item=self.create_text_output_item(text=content, id=str(uuid4())),
                    )
            else:
                # Fallback: retornar resultado como texto
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_text_output_item(text=str(result), id=str(uuid4())),
                )

        except Exception as e:
            error_msg = f"Error querying Genie Space: {str(e)}"
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_text_output_item(text=error_msg, id=str(uuid4())),
            )


#######################################################
# Inicializar Agent e configurar MLflow
#######################################################

mlflow.langchain.autolog()
AGENT = GenieBot(genie_agent=genie_agent)
mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testar Agent Localmente

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# from genie_bot_agent import AGENT

# Teste simples
input_example = {
    "input": [{"role": "user", "content": "Quantas verificações tivemos na area rodrigo silveira em 2025?"}]
}

result = AGENT.predict(input_example)
# print("Response:")
# print(result)

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
    DatabricksFunction
)
from pkg_resources import get_distribution

# TODO: Substituir pelos valores corretos
WAREHOUSE_ID = "4d81765d86b13aa1"  # ID do SQL Warehouse usado pelo Genie
CATALOG = "hs_franquia"
SCHEMA = "gold_crm"

# TODO: Adicionar todas as tabelas usadas pelo Genie Space
TABLES = [
    "d_acao_controle",
    "d_acao_data_inicio",
    "d_acao_prazo",
    "d_acao_responsavel",
    "d_acao_risco",
    "d_acao_status_acao",
    "d_acao_tipo_plano_acao",
    "d_acao_vencimento_acao",
    "d_pergunta_controle",
    "d_pergunta_resposta",
    "d_pergunta_risco",
    "d_pessoa_org_unit",
    "d_verificacao_data_verificacao",
    "d_verificacao_faixa_de_horario",
    "d_verificacao_localizacao",
    "f_acao",
    "f_pergunta",
    "f_pessoa",
    "f_verificacao",
    "filtro_vbm"
]

# TODO: Adicionar as functions usadas pelo agente
FUNCTIONS = [
    "hs_franquia.gold_connect_bot.get_subordinate_locations",
    "hs_franquia.gold_connect_bot.get_subordinate_org_units",
    "hs_franquia.gold_connect_bot.resolve_hierarchy_from_text",
    "hs_franquia.gold_connect_bot.resolve_location_from_text"
]

resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
    DatabricksGenieSpace(genie_space_id=GENIE_SPACE_ID),
    DatabricksSQLWarehouse(warehouse_id=WAREHOUSE_ID),
]

# Adicionar todas as tabelas como resources
for table in TABLES:
    resources.append(DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.{table}"))

for func in FUNCTIONS:
    resources.append(DatabricksFunction(function_name=func))

input_example = {
    "input": [{"role": "user", "content": "Quantas verificações em 2024?"}]
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="genie_gold_crm_bot",
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
    model_uri=f"runs:/{logged_agent_info.run_id}/genie_gold_crm_bot",
    input_data=input_example,
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registrar no Unity Catalog

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# Definir local do modelo no Unity Catalog
UC_MODEL_NAME = "hs_franquia.gold_connect_bot.genie_gold_crm_bot"

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
    tags={"endpointSource": "genie_gold_crm_bot"}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Próximos Passos
# MAGIC
# MAGIC 1. Teste o endpoint via AI Playground
# MAGIC 2. Valide as respostas do Genie
# MAGIC 3. Monitore traces no MLflow
# MAGIC 4. Adicione features incrementalmente conforme necessário
